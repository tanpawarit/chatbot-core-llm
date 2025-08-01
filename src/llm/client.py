from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Optional, Dict, Any
from src.config import config_manager
from src.models import Message, MessageRole, LLMModelConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseLLMClient:
    """Base class for scalable LLM clients"""
    
    def __init__(self, model_config: LLMModelConfig, client_name: str):
        self.model_config = model_config
        self.client_name = client_name
        self.openrouter_config = config_manager.get_openrouter_config()
        self._client: Optional[ChatOpenAI] = None
    
    @property
    def client(self) -> ChatOpenAI:
        if self._client is None:
            self._client = ChatOpenAI(
                model=self.model_config.model,
                openai_api_key=self.openrouter_config.api_key,
                openai_api_base=self.openrouter_config.base_url,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
            )
            logger.info(f"{self.client_name} initialized", 
                       model=self.model_config.model,
                       temperature=self.model_config.temperature)
        
        return self._client
    
    def _convert_to_langchain_messages(self, messages: List[Message]) -> List:
        """Convert our Message objects to LangChain message format"""
        langchain_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.USER:
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                langchain_messages.append(AIMessage(content=msg.content))
            elif msg.role == MessageRole.SYSTEM:
                langchain_messages.append(SystemMessage(content=msg.content))
        
        return langchain_messages


class ClassificationLLM(BaseLLMClient):
    """LLM specialized for event classification (JSON output)"""
    
    def __init__(self):
        config = config_manager.get_openrouter_config()
        super().__init__(config.classification, "ClassificationLLM")
    
    def classify_event(self, messages: List[Message]) -> str:
        """Generate JSON classification from messages"""
        try:
            langchain_messages = self._convert_to_langchain_messages(messages)
            
            logger.info("Classifying event", 
                       message_count=len(messages),
                       model=self.model_config.model)
            
            response = self.client.invoke(langchain_messages)
            
            logger.info("Event classification completed", 
                       response_length=len(response.content),
                       model=self.model_config.model)
            
            return response.content
            
        except Exception as e:
            logger.error("Failed to classify event", 
                        error=str(e),
                        model=self.model_config.model)
            raise


class ResponseLLM(BaseLLMClient):
    """LLM specialized for chat response generation (text output)"""
    
    def __init__(self):
        config = config_manager.get_openrouter_config()
        super().__init__(config.response, "ResponseLLM")
    
    def generate_response(self, messages: List[Message]) -> str:
        """Generate chat response from conversation messages"""
        try:
            langchain_messages = self._convert_to_langchain_messages(messages)
            
            logger.info("Generating chat response", 
                       message_count=len(messages),
                       model=self.model_config.model)
            
            response = self.client.invoke(langchain_messages)
            
            logger.info("Chat response generated", 
                       response_length=len(response.content),
                       model=self.model_config.model)
            
            return response.content
            
        except Exception as e:
            logger.error("Failed to generate chat response", 
                        error=str(e),
                        model=self.model_config.model)
            raise


class LLMClientManager:
    """Manager for scalable LLM clients"""
    
    def __init__(self):
        self._clients: Dict[str, BaseLLMClient] = {}
        self._initialize_default_clients()
    
    def _initialize_default_clients(self):
        """Initialize default classification and response clients"""
        self._clients['classification'] = ClassificationLLM()
        self._clients['response'] = ResponseLLM()
    
    def get_client(self, client_type: str) -> BaseLLMClient:
        """Get client by type (classification, response, or custom)"""
        if client_type not in self._clients:
            raise ValueError(f"Unknown client type: {client_type}")
        return self._clients[client_type]
    
    def add_custom_client(self, name: str, model_config: LLMModelConfig, client_class: type = None):
        """Add custom LLM client for scaling"""
        if client_class is None:
            client_class = BaseLLMClient
        
        self._clients[name] = client_class(model_config, f"CustomLLM-{name}")
        logger.info(f"Added custom LLM client: {name}")
    
    @property
    def classification_llm(self) -> ClassificationLLM:
        return self._clients['classification']
    
    @property
    def response_llm(self) -> ResponseLLM:
        return self._clients['response']


# Global instances
llm_manager = LLMClientManager()
classification_llm = llm_manager.classification_llm
response_llm = llm_manager.response_llm