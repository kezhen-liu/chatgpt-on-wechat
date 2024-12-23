"""
Google gemini bot

@author zhayujie
@Date 2023/12/15
"""
# encoding:utf-8

from bot.bot import Bot
import google.generativeai as genai
from bot.session_manager import SessionManager
from bridge.context import ContextType, Context
from bridge.reply import Reply, ReplyType
from common.log import logger
from common.token_bucket import TokenBucket
from config import conf
from bot.chatgpt.chat_gpt_session import ChatGPTSession
from bot.baidu.baidu_wenxin_session import BaiduWenxinSession
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class GeminiBotRateLimitError(Exception):
    pass

# OpenAI对话模型API (可用)
class GoogleGeminiBot(Bot):

    def __init__(self):
        super().__init__()
        self.api_key = conf().get("gemini_api_key")
        # 复用chatGPT的token计算方式
        self.sessions = SessionManager(ChatGPTSession, model=conf().get("model") or "gpt-3.5-turbo")
        self.model = conf().get("model") or "gemini-pro"
        if self.model == "gemini":
            self.model = "gemini-pro"
        if conf().get("rate_limit_chatgpt"):
            self.tb4gemini = TokenBucket(conf().get("rate_limit_chatgpt", 15), timeout = 5.0)
        self.grounding_prefix = None
        if conf().get("gemini_grounding_prefix"):
            self.grounding_prefix = conf().get("gemini_grounding_prefix", "!search")
    def reply(self, query, context: Context = None) -> Reply:
        try:
            if context.type != ContextType.TEXT:
                logger.warn(f"[Gemini] Unsupported message type, type={context.type}")
                return Reply(ReplyType.TEXT, None)
            if conf().get("rate_limit_chatgpt") and not self.tb4gemini.get_token():
                raise GeminiBotRateLimitError("RateLimitError: rate limit exceeded")
            logger.info(f"[Gemini] query={query}")
            session_id = context["session_id"]
            session = self.sessions.session_query(query, session_id)
            gemini_messages = self._convert_to_gemini_messages(self.filter_messages(session.messages))
            logger.debug(f"[Gemini] messages={gemini_messages}")
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            
            # 添加安全设置
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            use_grounding = False
            if self.grounding_prefix is not None and gemini_messages[-1] and gemini_messages[-1]["parts"][0]["text"].startswith(self.grounding_prefix):
                gemini_messages[-1]["parts"][0]["text"] = gemini_messages[-1]["parts"][0]["text"].replace(self.grounding_prefix, "", 1).strip()
                use_grounding = True
                logger.info("[Gemini] grounding enabled")

            # 生成回复，包含安全设置
            response = model.generate_content(
                gemini_messages,
                safety_settings=safety_settings,
                tools = 'google_search_retrieval' if use_grounding else None
            )
            if response.candidates and response.candidates[0].content:
                reply_text = response.candidates[0].content.parts[0].text
                logger.info(f"[Gemini] reply={reply_text}")
                self.sessions.session_reply(reply_text, session_id)
                return Reply(ReplyType.TEXT, reply_text)
            else:
                # 没有有效响应内容，可能内容被屏蔽，输出安全评分
                logger.warning("[Gemini] No valid response generated. Checking safety ratings.")
                if hasattr(response, 'candidates') and response.candidates:
                    for rating in response.candidates[0].safety_ratings:
                        logger.warning(f"Safety rating: {rating.category} - {rating.probability}")
                error_message = "No valid response generated due to safety constraints."
                self.sessions.session_reply(error_message, session_id)
                return Reply(ReplyType.ERROR, error_message)
                    
        except Exception as e:
            if isinstance(e, GeminiBotRateLimitError):
                logger.error(f"[Gemini] Error generating response: {str(e)}", exc_info=True)
                return Reply(ReplyType.TEXT, "提问太快啦，请休息一下再问我吧")
            logger.error(f"[Gemini] Error generating response: {str(e)}", exc_info=True)
            error_message = "Failed to invoke [Gemini] api!"
            self.sessions.session_reply(error_message, session_id)
            return Reply(ReplyType.ERROR, error_message)
            
    def _convert_to_gemini_messages(self, messages: list):
        res = []
        for msg in messages:
            if msg.get("role") == "user":
                role = "user"
            elif msg.get("role") == "assistant":
                role = "model"
            elif msg.get("role") == "system":
                role = "user"
            else:
                continue
            res.append({
                "role": role,
                "parts": [{"text": msg.get("content")}]
            })
        return res

    @staticmethod
    def filter_messages(messages: list):
        res = []
        turn = "user"
        if not messages:
            return res
        for i in range(len(messages) - 1, -1, -1):
            message = messages[i]
            role = message.get("role")
            if role == "system":
                res.insert(0, message)
                continue
            if role != turn:
                continue
            res.insert(0, message)
            if turn == "user":
                turn = "assistant"
            elif turn == "assistant":
                turn = "user"
        return res
