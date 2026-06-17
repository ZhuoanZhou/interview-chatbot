"""
Token usage tracking utilities for monitoring LLM token consumption across agents.
"""
import json
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path


class TokenUsageTracker:
    """
    Tracks token usage across multiple agents in a session.

    This tracker aggregates token usage statistics for all agents except UserAgent,
    and provides methods to save snapshots for offline analysis.
    """

    def __init__(self, session_id: str, user_id: str):
        """
        Initialize the token usage tracker.

        Args:
            session_id: Unique identifier for the session
            user_id: User identifier for organizing logs
        """
        self.session_id = session_id
        self.user_id = user_id
        self.start_time = datetime.now()

        # Track usage per agent: {agent_name: {prompt_tokens, completion_tokens, total_tokens, call_count}}
        self.agent_usage: Dict[str, Dict[str, int]] = {}

        # Track per-turn snapshots: [{turn, timestamp, agent, prompt_tokens, completion_tokens, total_tokens}]
        self.turn_snapshots = []

        # Global counters
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_calls = 0

    def record_usage(self, agent_name: str, turn: int, usage: Dict[str, int]):
        """
        Record token usage for a specific agent call.

        Args:
            agent_name: Name of the agent making the call
            turn: Current turn number in the session
            usage: Dictionary with keys 'prompt_tokens', 'completion_tokens', 'total_tokens'
        """
        # Skip UserAgent as requested
        if agent_name.lower() == "useragent":
            return

        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)

        # Initialize agent tracking if needed
        if agent_name not in self.agent_usage:
            self.agent_usage[agent_name] = {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'call_count': 0
            }

        # Update agent-specific counters
        self.agent_usage[agent_name]['prompt_tokens'] += prompt_tokens
        self.agent_usage[agent_name]['completion_tokens'] += completion_tokens
        self.agent_usage[agent_name]['total_tokens'] += total_tokens
        self.agent_usage[agent_name]['call_count'] += 1

        # Update global counters
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.total_calls += 1

        # Record per-turn snapshot
        self.turn_snapshots.append({
            'turn': turn,
            'timestamp': datetime.now().isoformat(),
            'agent': agent_name,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens
        })

    def get_summary(self) -> Dict:
        """
        Get a summary of all token usage.

        Returns:
            Dictionary containing session summary and per-agent statistics
        """
        return {
            'session_info': {
                'session_id': self.session_id,
                'user_id': self.user_id,
                'start_time': self.start_time.isoformat(),
                'current_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - self.start_time).total_seconds()
            },
            'total_usage': {
                'prompt_tokens': self.total_prompt_tokens,
                'completion_tokens': self.total_completion_tokens,
                'total_tokens': self.total_tokens,
                'total_calls': self.total_calls
            },
            'agent_breakdown': self.agent_usage,
            'turn_history': self.turn_snapshots
        }

    def save_snapshot(self, output_dir: Optional[str] = None):
        """
        Save a JSON snapshot of current token usage statistics.

        Args:
            output_dir: Directory to save the snapshot. If None, uses default location
                       based on user_id and session_id
        """
        if output_dir is None:
            output_dir = f"logs/{self.user_id}/statistics/session_{self.session_id}"

        # Create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        filename = f"token_usage_statistics.json"
        filepath = Path(output_dir) / filename

        # Save summary to JSON file
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        return str(filepath)

    def save_final_summary(self, output_dir: Optional[str] = None):
        """
        Save the final session summary.

        Args:
            output_dir: Directory to save the summary. If None, uses default location
        """
        if output_dir is None:
            output_dir = f"logs/{self.user_id}/statistics/session_{self.session_id}"

        # Create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        filename = "token_usage_statistics_final.json"
        filepath = Path(output_dir) / filename

        # Save summary to JSON file
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        return str(filepath)
