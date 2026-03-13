"""
AstrBot 插件：Grok 联网搜索

通过 Grok API 进行实时联网搜索，支持：
- /grok 指令
- LLM Tool (grok_web_search)
- Skill 脚本动态安装
"""

import shutil
import tempfile
import zipfile
from pathlib import Path
import re

import aiohttp
import asyncio
import json
import time

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.core.star.filter.command import GreedyStr
from astrbot.api.star import Context, Star
from astrbot.core.message.components import Image
from astrbot.core.utils.io import download_image_by_url, file_to_base64
from astrbot.core.utils.quoted_message.chain_parser import (
    _extract_image_refs_from_component_chain,
    _extract_text_from_component_chain,
)
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.provider.func_tool_manager import FunctionToolManager

from .grok_client import (
    DEFAULT_JSON_SYSTEM_PROMPT,
    grok_search,
    normalize_api_key,
    normalize_base_url,
    parse_json_config,
)

PLUGIN_NAME = "astrbot_plugin_grok_web_search"


def _fmt_tokens(n: int) -> str:
    """将 token 数量格式化为简短形式，如 1m2k、3.5k、800。"""
    if n >= 1_000_000:
        m, remain = divmod(n, 1_000_000)
        k = remain // 1_000
        return f"{m}m{k}k" if k else f"{m}m"
    if n >= 1_000:
        k, remain = divmod(n, 1_000)
        h = remain // 100
        return f"{k}.{h}k" if h else f"{k}k"
    return str(n)


class GrokSearchPlugin(Star):
    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        self.config = config or {}
        self._session: aiohttp.ClientSession | None = None

    async def _extract_content_from_event(
        self, event: AstrMessageEvent
    ) -> tuple[str | None, list[str]]:
        """Extract text and images from the user's message.

        Reuses AstrBot core's chain_parser for text/image extraction from
        Reply, Node, Nodes, Forward, etc.

        Returns:
            A tuple of (text, images):
            - text: extracted text from the message chain (or None)
            - images: list of base64-encoded image strings (without prefix)
        """
        chain = event.get_messages()

        # 使用本体的 chain_parser 提取文本（处理 Reply/Node/Nodes/Forward）
        text = _extract_text_from_component_chain(chain)

        # 使用本体的 chain_parser 提取图片引用，再转为 base64
        image_refs = _extract_image_refs_from_component_chain(chain)
        images: list[str] = []
        seen: set[str] = set()

        # 提取消息链顶层的 Image 组件并转为 base64
        for comp in chain:
            if isinstance(comp, Image):
                try:
                    b64 = await comp.convert_to_base64()
                    if b64 and b64 not in seen:
                        seen.add(b64)
                        images.append(b64)
                except Exception as e:
                    logger.warning(
                        f"[{PLUGIN_NAME}] Failed to convert image to base64: {e}"
                    )

        # 将嵌套组件中的图片引用（URL/路径）转为 base64
        for ref in image_refs:
            try:
                img = Image.fromURL(ref)
                b64 = await img.convert_to_base64()
                if b64 and b64 not in seen:
                    seen.add(b64)
                    images.append(b64)
            except Exception as e:
                logger.warning(
                    f"[{PLUGIN_NAME}] Failed to convert image ref to base64: {e}"
                )

        return text, images

    async def initialize(self):
        """插件初始化：验证配置并处理 Skill 安装"""
        # 如果启用使用 AstrBot 自带供应商，则推迟创建会话和 Skill 安装
        if self.config.get("use_builtin_provider", False):
            logger.info(
                f"[{PLUGIN_NAME}] use_builtin_provider enabled, delaying full initialization until AstrBot is loaded"
            )
            return

        # 仅在使用外部 HTTP 客户端时校验 base_url/api_key
        await self._validate_config()

        # 根据配置决定是否创建复用的 HTTP 会话
        if self.config.get("reuse_session", False):
            self._session = aiohttp.ClientSession()

        # 首次安装：将插件目录的 skill 移动到持久化目录
        self._migrate_skill_to_persistent()

        if self.config.get("enable_skill", False):
            self._install_skill()
        else:
            self._uninstall_skill()

    async def _validate_config(self):
        """验证必要配置，并通过 v1/models 接口检查连通性"""
        base_url = normalize_base_url(self.config.get("base_url", ""))
        api_key = normalize_api_key(self.config.get("api_key", ""))
        if not base_url:
            logger.warning(
                f"[{PLUGIN_NAME}] 缺少 base_url 配置，请在插件设置中填写 Grok API 端点"
            )
            return
        if not api_key:
            logger.warning(
                f"[{PLUGIN_NAME}] 缺少 api_key 配置，请在插件设置中填写 API 密钥"
            )
            return

        # 通过 v1/models 接口验证连通性和密钥有效性
        models_url = f"{base_url}/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        extra_headers = self._parse_json_config("extra_headers")
        if extra_headers:
            protected = {"authorization", "content-type"}
            for key, value in extra_headers.items():
                if str(key).lower() not in protected:
                    headers[str(key)] = str(value)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    models_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 401:
                        logger.warning(
                            f"[{PLUGIN_NAME}] API 密钥无效（401），请检查 api_key 配置"
                        )
                    elif resp.status == 403:
                        logger.warning(
                            f"[{PLUGIN_NAME}] API 密钥权限不足（403），请检查 api_key 权限"
                        )
                    elif resp.status == 404:
                        logger.warning(
                            f"[{PLUGIN_NAME}] v1/models 端点不存在（404），请检查 base_url 配置是否正确"
                        )
                    elif resp.status != 200:
                        logger.warning(
                            f"[{PLUGIN_NAME}] API 连通性检查返回 HTTP {resp.status}，请确认配置"
                        )
                    else:
                        logger.info(f"[{PLUGIN_NAME}] API 连通性检查通过")
        except aiohttp.ClientError as e:
            logger.warning(
                f"[{PLUGIN_NAME}] API 连通性检查失败（网络错误）: {e}，请检查 base_url 配置"
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[{PLUGIN_NAME}] API 连通性检查超时，请检查 base_url 是否可达"
            )

    def _get_skill_manager(self):
        """获取 SkillManager 实例（延迟导入）"""
        if hasattr(self, "_skill_mgr"):
            return self._skill_mgr
        try:
            from astrbot.core.skills import SkillManager

            self._skill_mgr = SkillManager()
        except ImportError:
            self._skill_mgr = None
        return self._skill_mgr

    def _get_plugin_data_path(self) -> Path:
        """获取插件持久化数据目录"""
        try:
            from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

            plugin_data_root = Path(get_astrbot_plugin_data_path())
        except ImportError:
            # 回退到相对路径
            plugin_data_root = Path(__file__).parent.parent.parent / "plugin_data"

        # 插件专属目录
        plugin_data_dir = plugin_data_root / PLUGIN_NAME
        plugin_data_dir.mkdir(parents=True, exist_ok=True)
        return plugin_data_dir

    def _get_skill_persistent_path(self) -> Path:
        """获取 Skill 持久化存储路径"""
        return self._get_plugin_data_path() / "skill"

    def _migrate_skill_to_persistent(self):
        """首次安装：将插件目录的 skill 复制到持久化目录"""
        source_dir = Path(__file__).parent / "skill"
        persistent_dir = self._get_skill_persistent_path()

        if source_dir.exists() and not persistent_dir.exists():
            try:
                shutil.copytree(source_dir, persistent_dir, symlinks=True)
                logger.info(
                    f"[{PLUGIN_NAME}] Skill 已复制到持久化目录: {persistent_dir}"
                )
            except Exception as e:
                logger.error(f"[{PLUGIN_NAME}] Skill 复制到持久化目录失败: {e}")

    def _install_skill(self):
        """通过 SkillManager 安装 Skill（打包为 zip 后调用官方接口）"""
        source_dir = self._get_skill_persistent_path()

        if not source_dir.exists():
            logger.error(f"[{PLUGIN_NAME}] Skill 持久化目录不存在: {source_dir}")
            return

        if source_dir.is_symlink():
            logger.error(
                f"[{PLUGIN_NAME}] Skill 源目录是 symlink，拒绝安装: {source_dir}"
            )
            return

        skill_mgr = self._get_skill_manager()
        if not skill_mgr:
            logger.error(f"[{PLUGIN_NAME}] SkillManager 不可用，无法安装 Skill")
            return

        tmp_zip = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp_zip = Path(tmp.name)

            with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in source_dir.rglob("*"):
                    if file.is_file():
                        arcname = f"grok-search/{file.relative_to(source_dir)}"
                        zf.write(file, arcname)

            skill_mgr.install_skill_from_zip(str(tmp_zip), overwrite=True)
            logger.info(f"[{PLUGIN_NAME}] Skill 已通过 SkillManager 安装并激活")
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] Skill 安装失败: {e}")
        finally:
            if tmp_zip:
                tmp_zip.unlink(missing_ok=True)

    def _uninstall_skill(self):
        """通过 SkillManager 卸载 Skill"""
        skill_mgr = self._get_skill_manager()
        if not skill_mgr:
            logger.error(f"[{PLUGIN_NAME}] SkillManager 不可用，无法卸载 Skill")
            return

        try:
            skill_mgr.delete_skill("grok-search")
            logger.info(f"[{PLUGIN_NAME}] Skill 已通过 SkillManager 卸载")
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] Skill 卸载失败: {e}")

    def _parse_json_config(self, key: str) -> dict:
        """解析 JSON 格式的配置项"""
        value = self.config.get(key, "")
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            result, error = parse_json_config(value)
            if error:
                logger.warning(f"[{PLUGIN_NAME}] {key} {error}")
            return result
        return {}

    async def _do_search(
        self,
        query: str,
        system_prompt: str | None = None,
        use_retry: bool = False,
        images: list[str] | None = None,
    ) -> dict:
        """Execute a search.

        Args:
            query: Search query content
            system_prompt: Custom system prompt, uses default when None
            use_retry: Whether to enable retry (command invocation only)
            images: Optional list of base64-encoded images for multimodal queries
        """
        # 安全解析 timeout 配置
        try:
            timeout_val = self.config.get("timeout_seconds", 60)
            timeout = float(timeout_val) if timeout_val is not None else 60.0
            if timeout <= 0:
                timeout = 60.0
        except (ValueError, TypeError):
            timeout = 60.0

        # 安全解析 thinking_budget 配置
        try:
            thinking_budget_val = self.config.get("thinking_budget", 32000)
            thinking_budget = (
                int(thinking_budget_val) if thinking_budget_val is not None else 32000
            )
            if thinking_budget < 0:
                thinking_budget = 32000
        except (ValueError, TypeError):
            thinking_budget = 32000

        # 重试配置（仅指令调用时使用）
        max_retries = 0
        retry_delay = 1.0
        retryable_status_codes = None
        if use_retry:
            max_retries = self.config.get("max_retries", 3)
            retry_delay = self.config.get("retry_delay", 1.0)

            # 解析可重试状态码（直接从 list 类型配置获取）
            retryable_codes = self.config.get("retryable_status_codes", [])
            if retryable_codes and isinstance(retryable_codes, list):
                retryable_status_codes = set(retryable_codes)

        # 自定义系统提示词
        custom_prompt = self.config.get("custom_system_prompt", "")
        if custom_prompt and isinstance(custom_prompt, str) and custom_prompt.strip():
            # 如果有自定义提示词且没有传入其他提示词，使用配置中的自定义提示词
            if system_prompt is None:
                system_prompt = custom_prompt.strip()
        # 如果仍然没有系统提示词，使用默认的 JSON 系统提示词
        if system_prompt is None:
            system_prompt = DEFAULT_JSON_SYSTEM_PROMPT
        # 如果启用了使用 AstrBot 自带供应商，通过 AstrBot provider 接口调用
        if self.config.get("use_builtin_provider", False):
            attempts = 0
            last_exc = None
            started = time.time()
            while True:
                try:
                    # 严格按配置获取 provider
                    configured_provider_id = self.config.get("provider", "")
                    if not configured_provider_id:
                        return {
                            "ok": False,
                            "error": "启用了内置供应商但未选择供应商，请在插件设置中选择一个 LLM 供应商",
                        }
                    prov = self.context.get_provider_by_id(configured_provider_id)
                    if not prov:
                        return {
                            "ok": False,
                            "error": f"未找到配置的供应商: {configured_provider_id}",
                        }

                    provider_id = prov.meta().id

                    # 将 base64 图片转为内置供应商的 image_urls 格式
                    image_urls = (
                        [f"base64://{img}" for img in images] if images else None
                    )

                    llm_resp = await self.context.llm_generate(
                        chat_provider_id=provider_id,
                        prompt=query,
                        system_prompt=system_prompt,
                        image_urls=image_urls,
                    )

                    text = llm_resp.completion_text or ""
                    usage = {}
                    if llm_resp.usage:
                        usage = {
                            "prompt_tokens": llm_resp.usage.input,
                            "completion_tokens": llm_resp.usage.output,
                            "total_tokens": llm_resp.usage.total,
                        }

                    # 尝试解析 JSON 格式响应
                    parsed = self._try_parse_json_response(text)
                    if parsed is not None:
                        content = str(parsed.get("content", ""))
                        raw_sources = parsed.get("sources", [])
                        sources = self._normalize_sources(raw_sources)
                        return {
                            "ok": True,
                            "content": content,
                            "sources": sources,
                            "elapsed_ms": int((time.time() - started) * 1000),
                            "retries": attempts,
                            "usage": usage,
                            "raw": "",
                        }

                    # JSON 解析失败，降级处理：提取纯文本和 URL
                    logger.warning(
                        f"[{PLUGIN_NAME}] 内置供应商返回非 JSON 格式，使用降级处理"
                    )

                    # 检测典型错误模式，避免将错误文案误判为成功
                    text_lower = text.lower()
                    error_patterns = [
                        "rate limit",
                        "too many requests",
                        "quota exceeded",
                        "authentication failed",
                        "invalid api key",
                        "unauthorized",
                        "service unavailable",
                        "internal server error",
                        "timeout",
                        "connection refused",
                    ]
                    is_error_response = any(p in text_lower for p in error_patterns)

                    if not text.strip() or is_error_response:
                        error_msg = (
                            "提供商返回空响应"
                            if not text.strip()
                            else f"提供商返回错误: {text[:200]}"
                        )
                        return {
                            "ok": False,
                            "error": error_msg,
                            "content": "",
                            "sources": [],
                            "elapsed_ms": int((time.time() - started) * 1000),
                            "retries": attempts,
                            "usage": usage,
                            "raw": text[:500] if text else "",
                        }

                    sources = self._extract_sources_from_text(text)
                    return {
                        "ok": True,
                        "content": text,
                        "sources": sources,
                        "elapsed_ms": int((time.time() - started) * 1000),
                        "retries": attempts,
                        "usage": usage,
                        "raw": text,
                    }

                except Exception as e:
                    last_exc = e
                    attempts += 1
                    if not use_retry or attempts > max_retries:
                        return {"ok": False, "error": str(last_exc)}
                    await asyncio.sleep(retry_delay * attempts)

        # 否则使用 HTTP 客户端向外部 Grok API 发起请求
        try:
            result = await grok_search(
                query=query,
                base_url=self.config.get("base_url", ""),
                api_key=self.config.get("api_key", ""),
                model=self.config.get("model", "grok-4-fast"),
                timeout=timeout,
                enable_thinking=self.config.get("enable_thinking", True),
                thinking_budget=thinking_budget,
                extra_body=self._parse_json_config("extra_body"),
                extra_headers=self._parse_json_config("extra_headers"),
                session=self._session,
                system_prompt=system_prompt,
                max_retries=max_retries,
                retry_delay=retry_delay,
                retryable_status_codes=retryable_status_codes,
                images=images,
            )
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] API 调用异常: {e}")
            return {"ok": False, "error": f"API 调用异常: {e}"}

        if not result.get("ok"):
            logger.warning(
                f"[{PLUGIN_NAME}] API 调用失败: {result.get('error', '未知错误')}"
            )
        return result

    def _format_result(self, result: dict) -> str:
        """格式化搜索结果为用户友好的消息"""
        if not result.get("ok"):
            error = result.get("error", "未知错误")
            return f"搜索失败: {error}"

        content = result.get("content", "")
        sources = result.get("sources", [])
        elapsed = result.get("elapsed_ms", 0) / 1000

        show_sources = self.config.get("show_sources", False)
        max_sources = self.config.get("max_sources", 5)

        lines = [content]

        if show_sources and sources:
            if max_sources > 0:
                sources = sources[:max_sources]
            lines.append("\n来源:")
            for i, src in enumerate(sources, 1):
                url = src.get("url", "")
                title = src.get("title", "")
                if title:
                    lines.append(f"  {i}. {title}\n     {url}")
                else:
                    lines.append(f"  {i}. {url}")

        # 显示耗时、重试次数和 token 用量
        retry_info = ""
        retries = result.get("retries", 0)
        if retries > 0:
            retry_info = f"，重试 {retries} 次"

        token_info = ""
        usage = result.get("usage") or {}
        total_tokens = usage.get("total_tokens", 0)
        if total_tokens:
            token_info = f"，tokens: {_fmt_tokens(total_tokens)}"

        lines.append(f"\n(耗时: {elapsed:.1f}s{retry_info}{token_info})")

        return "\n".join(lines)

    def _format_result_for_llm(self, result: dict) -> str:
        """格式化搜索结果供 LLM 使用（纯文本，无 Markdown）"""
        if not result.get("ok"):
            error = result.get("error", "未知错误")
            raw = result.get("raw", "")
            return f"搜索失败: {error}\n{raw}"

        content = result.get("content", "")
        sources = result.get("sources", [])

        show_sources = self.config.get("show_sources", False)
        max_sources = self.config.get("max_sources", 5)

        lines = [f"搜索结果:\n{content}"]

        if show_sources and sources:
            if max_sources > 0:
                sources = sources[:max_sources]
            lines.append("\n参考来源:")
            for i, src in enumerate(sources, 1):
                url = src.get("url", "")
                title = src.get("title", "")
                snippet = src.get("snippet", "")
                if title:
                    lines.append(f"  {i}. {title}")
                    lines.append(f"     {url}")
                else:
                    lines.append(f"  {i}. {url}")
                if snippet:
                    lines.append(f"     {snippet}")

        # 提示主 LLM 使用纯文本格式回复用户
        lines.append("\n[提示: 请使用纯文本格式回复用户，不要使用 Markdown 格式]")

        return "\n".join(lines)

    def _try_parse_json_response(self, text: str) -> dict | None:
        """尝试解析 JSON 响应，支持多种格式

        支持的格式：
        1. 纯 JSON 对象
        2. Markdown 代码块包裹的 JSON
        3. 混合文本中的 JSON（支持嵌套结构）
        """

        if not text or not text.strip():
            return None

        text = text.strip()

        # 尝试直接解析
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        # 尝试提取 Markdown 代码块中的 JSON
        code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        matches = re.findall(code_block_pattern, text)
        for match in matches:
            try:
                parsed = json.loads(match.strip())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        # 使用 JSONDecoder.raw_decode 从每个 { 起点尝试解码（支持嵌套结构）
        decoder = json.JSONDecoder()
        start_idx = 0
        max_attempts = 10  # 限制尝试次数

        while start_idx < len(text) and max_attempts > 0:
            brace_pos = text.find("{", start_idx)
            if brace_pos == -1:
                break

            try:
                parsed, end_idx = decoder.raw_decode(text, idx=brace_pos)
                if isinstance(parsed, dict) and (
                    "content" in parsed or "sources" in parsed
                ):
                    return parsed
                start_idx = end_idx
            except json.JSONDecodeError:
                start_idx = brace_pos + 1

            max_attempts -= 1

        return None

    def _normalize_sources(self, raw_sources: list) -> list[dict[str, str]]:
        """归一化 sources 结构，仅允许 http/https 协议"""
        from urllib.parse import urlparse

        sources = []
        if isinstance(raw_sources, list):
            for item in raw_sources:
                if isinstance(item, dict) and item.get("url"):
                    url = str(item.get("url", ""))
                    # URL 协议白名单校验
                    try:
                        parsed = urlparse(url)
                        if parsed.scheme not in ("http", "https"):
                            continue
                        # 限制长度和过滤控制字符
                        if len(url) > 2048 or any(ord(c) < 32 for c in url):
                            continue
                    except Exception:
                        continue

                    sources.append(
                        {
                            "url": url,
                            "title": str(item.get("title") or ""),
                            "snippet": str(item.get("snippet") or ""),
                        }
                    )
        return sources

    def _extract_sources_from_text(self, text: str) -> list[dict[str, str]]:
        """从文本中提取 URL 作为来源，仅允许 http/https 协议"""
        from urllib.parse import urlparse

        sources = []
        url_pattern = r"https://[^\s)\]}>\"']+|http://[^\s)\]}>\"']+"
        seen: set[str] = set()

        for match in re.finditer(url_pattern, text):
            url = match.group().rstrip(".,;:!?\"'")
            if not url or url in seen:
                continue
            # URL 校验
            try:
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https"):
                    continue
                if len(url) > 2048 or any(ord(c) < 32 for c in url):
                    continue
            except Exception:
                continue

            seen.add(url)
            sources.append({"url": url, "title": "", "snippet": ""})

        return sources

    def _help_text(self) -> str:
        """返回帮助文本"""
        use_builtin = self.config.get("use_builtin_provider", False)
        mode = "AstrBot 自带" if use_builtin else "自定义"
        provider_id = (
            (self.config.get("provider", "") or "未配置")
            if use_builtin
            else (self.config.get("base_url", "") or "未配置")
        )
        model = (
            "由供应商决定"
            if use_builtin
            else (self.config.get("model", "grok-4-fast") or "默认")
        )
        has_custom_prompt = bool(
            (self.config.get("custom_system_prompt", "") or "").strip()
        )
        if has_custom_prompt:
            prompt_info = "自定义"
        else:
            prompt_info = "内置中文（/grok 指令）/ 内置英文 JSON（LLM Tool）"

        return (
            "Grok 联网搜索\n"
            "\n"
            "用法:\n"
            "  /grok help           显示此帮助\n"
            "  /grok <搜索内容>     执行联网搜索\n"
            "\n"
            "示例:\n"
            "  /grok Python 3.12 有什么新特性\n"
            "  /grok 最新的 AI 新闻\n"
            "  /grok React 19 发布了吗\n"
            "\n"
            "调用方式:\n"
            "  - /grok 指令：直接搜索并返回结果\n"
            "  - LLM Tool：模型自动调用 grok_web_search\n"
            "\n"
            f"当前配置:\n"
            f"  供应商来源: {mode}\n"
            f"  供应商: {provider_id}\n"
            f"  模型: {model}\n"
            f"  系统提示词: {prompt_info}"
        )

    @filter.command("grok")
    async def grok_cmd(self, event: AstrMessageEvent, query: GreedyStr = ""):
        """执行 Grok 搜索

        用法: /grok <搜索内容>
        """
        # 提取消息中的文本和图片（包括引用消息/转发消息）
        extra_text, images = await self._extract_content_from_event(event)
        if images:
            logger.info(
                f"[{PLUGIN_NAME}] /grok command: extracted {len(images)} image(s) from message"
            )

        # 仅在明确输入 help 时显示帮助
        if query.strip().lower() == "help":
            yield event.plain_result(self._help_text())
            return

        # 无查询文本但有图片或引用内容时，继续搜索
        has_content = bool(images) or bool(extra_text)
        if not query.strip() and not has_content:
            yield event.plain_result(self._help_text())
            return

        # 将引用/转发消息中提取的文本拼接到查询前面作为上下文
        if extra_text:
            if query.strip():
                query = f"[Referenced message content]\n{extra_text}\n\n[User query]\n{query}"
            else:
                query = extra_text

        # 仅有图片无文本时，使用默认提示词
        if not query.strip() and images:
            query = "请搜索这张图片的内容"

        # 优先使用自定义提示词，未设置则使用内置提示词（英文指令 + JSON 格式 + 中文回复）
        custom_prompt = self.config.get("custom_system_prompt", "")
        if custom_prompt and isinstance(custom_prompt, str) and custom_prompt.strip():
            cmd_system_prompt = custom_prompt.strip()
        else:
            cmd_system_prompt = (
                "You are a web research assistant. Use live web search/browsing when answering. "
                "Return ONLY a single JSON object with keys: "
                "content (string), sources (array of objects with url/title/snippet when possible). "
                "Keep content concise and evidence-backed. "
                "IMPORTANT: Respond in Chinese. Do NOT use Markdown formatting in the content field - use plain text only. "
                "Keep proper nouns and names in their original language."
            )

        result = await self._do_search(
            query,
            system_prompt=cmd_system_prompt,
            use_retry=True,
            images=images or None,
        )
        event.should_call_llm(True)
        try:
            await event.send(MessageChain().message(self._format_result(result)))
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] 发送搜索结果失败: {e}")
            try:
                await event.send(
                    MessageChain().message("搜索完成，但消息发送失败，请重试。")
                )
            except Exception:
                pass

    @filter.llm_tool(name="grok_web_search")
    async def grok_tool(
        self,
        event: AstrMessageEvent,
        query: str,
        image_urls: str = "",
    ) -> str:
        """通过 Grok 进行实时联网搜索，获取最新信息和来源。支持传入图片进行多模态搜索。

        当需要搜索实时信息、最新新闻、API 版本、错误解决方案或验证过时/不确定信息时使用。
        如果用户消息中包含图片，也会自动提取图片进行多模态搜索。

        Args:
            query(string): 搜索查询内容，应该是清晰具体的问题或关键词
            image_urls(string): 可选，逗号分隔的图片 URL 或 base64:// 链接
        """
        # 收集图片：从 LLM 传入的 image_urls 参数 + 用户消息中提取
        images: list[str] = []

        # 1. 解析 LLM 传入的 image_urls
        if image_urls and isinstance(image_urls, str):
            for url in image_urls.split(","):
                url = url.strip()
                if not url:
                    continue
                if url.startswith("base64://"):
                    images.append(url.removeprefix("base64://"))
                elif url.startswith("http"):
                    # 下载并转为 base64
                    try:
                        file_path = await download_image_by_url(url)
                        b64 = file_to_base64(file_path)
                        b64 = b64.removeprefix("base64://")
                        if b64:
                            images.append(b64)
                    except Exception as e:
                        logger.warning(
                            f"[{PLUGIN_NAME}] Failed to download image from URL {url}: {e}"
                        )

        # 2. 从用户消息事件中自动提取内容
        extra_text, event_images = await self._extract_content_from_event(event)
        images.extend(event_images)

        # 将引用/转发消息中提取的文本拼接到查询前面作为上下文
        if extra_text:
            query = (
                f"[Referenced message content]\n{extra_text}\n\n[User query]\n{query}"
            )

        if images:
            logger.info(
                f"[{PLUGIN_NAME}] grok_web_search tool: processing with {len(images)} image(s)"
            )

        result = await self._do_search(query, use_retry=False, images=images or None)
        return self._format_result_for_llm(result)

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """在 LLM 请求时，如果启用了 Skill 则移除 grok_web_search 工具"""
        if not self.config.get("enable_skill", False):
            return

        tool_set = req.func_tool
        if isinstance(tool_set, FunctionToolManager):
            req.func_tool = tool_set.get_full_tool_set()
            tool_set = req.func_tool

        if tool_set:
            tool_set.remove_tool("grok_web_search")

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        """当 AstrBot 初始化完成后执行的钩子：在启用了自带供应商时完成插件的剩余初始化工作"""
        try:
            if not self.config.get("use_builtin_provider", False):
                return

            logger.info(f"[{PLUGIN_NAME}] AstrBot 已初始化，继续完成插件初始化")

            # 创建复用的 HTTP 会话（如果配置要求）
            if self.config.get("reuse_session", False) and (
                self._session is None or self._session.closed
            ):
                self._session = aiohttp.ClientSession()

            # 迁移并根据 enable_skill 安装或卸载 Skill
            self._migrate_skill_to_persistent()
            if self.config.get("enable_skill", False):
                self._install_skill()
            else:
                self._uninstall_skill()

        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] on_astrbot_loaded 处理失败: {e}")

    async def terminate(self):
        """插件销毁：关闭 HTTP 会话"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
