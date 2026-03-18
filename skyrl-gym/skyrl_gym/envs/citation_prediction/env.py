from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.citation_prediction.utils import compute_arxiv_score
from skyrl_gym.tools.search import call_search_api, SearchToolGroup
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from omegaconf import DictConfig
import json
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class CitationPredictionEnvConfig:
    log_requests: bool = False
    search_url: str = "http://127.0.0.1:8000/retrieve"
    topk: int = 3
    timeout: int = 30


class CitationPredictionEnv(BaseTextEnv):
    """
    Environment for citation prediction tasks.

    The model searches over an arxiv corpus to identify a masked citation,
    then submits its answer using <answer>arxiv_id</answer> tags.
    Reward = 1.0 if the answer matches the ground-truth arxiv ID (via
    compute_score exact match), 0.0 otherwise.

    A turn counter is appended to each observation so the model knows how
    many search attempts remain.  On the last turn the model is instructed
    to submit its answer.

    Unlike SearchEnv, this environment also tracks whether the correct
    paper appeared in any retrieval result (for metrics only).
    """

    def __init__(self, env_config: Union[CitationPredictionEnvConfig, DictConfig, dict], extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        reward_spec = extras["reward_spec"]
        if isinstance(reward_spec, str):
            reward_spec = json.loads(reward_spec)
        assert "ground_truth" in reward_spec, "ground_truth is required in reward_spec field"
        # Store as dict {"target": arxiv_id} for compatibility with compute_score
        self.ground_truth = reward_spec["ground_truth"]
        self.ground_truth_id = self.ground_truth["target"]
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 4

        # Convert plain dict to dataclass for uniform attribute access
        if isinstance(env_config, dict):
            env_config = CitationPredictionEnvConfig(**env_config)

        self.search_url = env_config.search_url
        self.topk = env_config.topk
        self.timeout = env_config.timeout
        self.log_requests = env_config.log_requests

        # We still register a SearchToolGroup so BaseTextEnv's tool infra is happy,
        # but we call the retriever API directly in _do_search() to access structured results.
        self.tool_group = SearchToolGroup(
            search_url=self.search_url,
            topk=self.topk,
            timeout=self.timeout,
            log_requests=self.log_requests,
        )
        self.init_tool_groups([self.tool_group])

        # Track whether the correct paper has been found in any retrieval
        self.found_correct_paper = False

        # Chat history
        self.chat_history: ConversationType = []

    def _parse_action(self, action: str) -> List[Optional[str]]:
        match = None
        if "<search>" in action and "</search>" in action:
            match = re.search(r"<search>(.*?)</search>", action, re.DOTALL)
        return [match.group(1)] if match else [None]

    def _do_search(self, query: str) -> str:
        """Call retriever API, check document IDs for reward, return formatted text for model."""
        api_response, error_msg = call_search_api(
            retrieval_service_url=self.search_url,
            query=query,
            topk=self.topk,
            timeout=self.timeout,
            log_requests=self.log_requests,
            session=self.tool_group.session,
        )

        if error_msg or not api_response:
            return "\n<information>" + json.dumps({"result": f"Search error: {error_msg}"}) + "</information>\n"

        raw_results = api_response.get("result", [])
        if not raw_results:
            return "\n<information>" + json.dumps({"result": "No search results found."}) + "</information>\n"

        # Check document IDs from the structured response
        for retrieval in raw_results:
            for doc_item in retrieval:
                doc_id = doc_item.get("document", {}).get("id", "")
                if doc_id == self.ground_truth_id:
                    self.found_correct_paper = True

        # Format text for the model (same as _passages2string)
        pretty_parts = []
        for retrieval in raw_results:
            formatted = ""
            for idx, doc_item in enumerate(retrieval):
                content = doc_item["document"]["contents"].strip()
                formatted += f"Doc {idx+1}: {content}\n"
            pretty_parts.append(formatted)
        final_result = "\n---\n".join(pretty_parts)

        return "\n<information>" + json.dumps({"result": final_result}) + "</information>\n"

    def _is_done(self, action: str) -> bool:
        if self.turns >= self.max_turns:
            return True
        return "<answer>" in action and "</answer>" in action

    def _get_reward(self, action: str, done: bool) -> float:
        if done:
            chat_history_str = "".join([item["content"] for item in self.chat_history])
            return compute_arxiv_score(chat_history_str, self.ground_truth)
        return 0.0

    def _validate_action(self, action: str):
        stop_tags = ["</search>", "</answer>"]
        for tag in stop_tags:
            if tag in action:
                assert action.split(tag, 1)[1] == "", (
                    f"{tag} detected in the response but it is not the last string generated. "
                    f"Use {stop_tags} as stop strings in the configuration."
                )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self._validate_action(action)
        self.chat_history.append({"role": "assistant", "content": action})

        error = None

        # Check if done (model answered or max turns reached)
        done = self._is_done(action)
        reward = self._get_reward(action, done)

        if done:
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})

        # Execute search if the model issued one
        observation = None
        query = self._parse_action(action)
        if query[0] is not None:
            try:
                observation = self._do_search(query[0])
            except Exception as e:
                error = str(e)
                observation = None

        # Wrap the observation properly as a message
        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            new_obs = {"role": "user", "content": error}
        else:
            new_obs = None

        # Append turn counter to the observation
        remaining = self.max_turns - self.turns
        if remaining > 1:
            turn_msg = f"\n\n{remaining} turns remaining."
        elif remaining == 1:
            turn_msg = "\n\nThis is your last turn. Submit your final answer now using answer tags. Do not search again."
        else:
            turn_msg = ""

        if new_obs:
            if turn_msg:
                new_obs["content"] += turn_msg
        else:
            if turn_msg:
                new_obs = {"role": "user", "content": turn_msg.strip()}

        info = {
            "tool_group": "SearchToolGroup",
            "tool_name": "search",
            "tool_input": query,
        }

        if new_obs:
            self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs] if new_obs else [],
            reward=reward,
            done=done,
            metadata=info,
        )

    def get_metrics(self) -> Dict[str, Any]:
        # Check if the model explicitly answered with <answer> tags
        chat_history_str = "".join([item["content"] for item in self.chat_history])
        answered = "<answer>" in chat_history_str and "</answer>" in chat_history_str
        return {
            "found_paper": int(self.found_correct_paper),
            "answered": int(answered),
            "num_turns": self.turns,
        }
