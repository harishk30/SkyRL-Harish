from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.citation_prediction.utils import extract_arxiv_ids, check_citation_match
from skyrl_gym.tools import SearchToolGroup
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from omegaconf import DictConfig
import re


@dataclass
class CitationPredictionEnvConfig:
    log_requests: bool = False
    search_url: str = "http://127.0.0.1:8000/retrieve"
    topk: int = 3
    timeout: int = 30


class CitationPredictionEnv(BaseTextEnv):
    """
    Environment for citation prediction tasks.

    The model searches over an arxiv corpus to identify a masked citation.
    Reward = 1.0 if the ground-truth arxiv ID appears in any retriever
    top-k results during the episode, 0.0 otherwise.
    No <answer> tags needed — reward is based purely on retrieval.
    """

    def __init__(self, env_config: Union[CitationPredictionEnvConfig, DictConfig, dict], extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth_id = extras["reward_spec"]["ground_truth"]["target"]
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 4

        # Convert plain dict to dataclass for uniform attribute access
        if isinstance(env_config, dict):
            env_config = CitationPredictionEnvConfig(**env_config)

        # Initialize the search tools (reuses SearchToolGroup from search env)
        self.tool_group = SearchToolGroup(
            search_url=env_config.search_url,
            topk=env_config.topk,
            timeout=env_config.timeout,
            log_requests=env_config.log_requests,
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

    def _execute_tool(self, tool_group_name: str, tool_name: str, tool_input: Any) -> str:
        tool_output = super()._execute_tool(tool_group_name, tool_name, tool_input)

        # Check if the ground-truth arxiv ID appears in the retriever output
        retrieved_ids = extract_arxiv_ids(tool_output)
        if check_citation_match(retrieved_ids, self.ground_truth_id):
            self.found_correct_paper = True

        return "\n<information>" + tool_output + "</information>\n"

    def _is_done(self, action: str) -> bool:
        if self.found_correct_paper:
            return True
        if self.turns >= self.max_turns:
            return True
        return False

    def _get_reward(self, done: bool) -> float:
        if done:
            return 1.0 if self.found_correct_paper else 0.0
        return 0.0

    def _validate_action(self, action: str):
        stop_tags = ["</search>"]
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

        # Try to execute the search tool before checking done,
        # so that a search on the final turn can still find the paper
        observation = None
        query = self._parse_action(action)
        if query[0] is not None:
            try:
                observation = self._execute_tool("SearchToolGroup", "search", query)
            except Exception as e:
                error = str(e)
                observation = None

        done = self._is_done(action)
        reward = self._get_reward(done)

        if done:
            # Still return the observation from this turn's search if available
            if observation:
                new_obs = {"role": "user", "content": observation}
                self.chat_history.append(new_obs)
                return BaseTextEnvStepOutput(
                    observations=[new_obs], reward=reward, done=done, metadata={}
                )
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})

        # Wrap the observation properly as a message
        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            new_obs = {"role": "user", "content": error}
        else:
            new_obs = None

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
        return {
            "found_paper": int(self.found_correct_paper),
            "num_turns": self.turns,
        }
