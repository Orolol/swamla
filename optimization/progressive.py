"""Progressive Training Scheduler for sequence length curriculum."""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ProgressivePhase:
    """A single phase in progressive training."""
    seq_len: int
    end_tokens: int  # End this phase after this many tokens (cumulative)
    batch_size: int


class ProgressiveScheduler:
    """
    Manages progressive training phases with sequence length curriculum.

    Automatically adjusts seq_len and batch_size based on tokens seen,
    keeping effective batch size (in tokens) approximately constant.

    Usage:
        scheduler = ProgressiveScheduler.from_schedule(
            "512:500M,1024:2B,2048:inf",
            base_batch_size=4,
            target_seq_len=2048
        )

        # In training loop:
        seq_len, batch_size = scheduler.get_current_config(total_tokens)
        if scheduler.check_phase_transition(total_tokens):
            print(f"Transitioning to seq_len={seq_len}")
    """

    def __init__(self, phases: List[ProgressivePhase]):
        """
        Args:
            phases: List of ProgressivePhase objects, ordered by end_tokens
        """
        if not phases:
            raise ValueError("At least one phase is required")

        # Sort by end_tokens
        self.phases = sorted(phases, key=lambda p: p.end_tokens)
        self.current_phase_idx = 0
        self._last_phase_idx = 0

    @classmethod
    def from_schedule(
        cls,
        schedule: str,
        base_batch_size: int,
        target_seq_len: int = 2048,
    ) -> "ProgressiveScheduler":
        """
        Create scheduler from schedule string.

        Args:
            schedule: Format "seq_len:tokens,seq_len:tokens,..."
                      e.g., "512:500M,1024:2B,2048:inf"
                      Tokens can use K, M, B suffixes or "inf"
            base_batch_size: Batch size for the target (longest) sequence length
            target_seq_len: The final/target sequence length

        Returns:
            ProgressiveScheduler instance
        """
        phases = []

        # Calculate target tokens per batch (for constant effective batch)
        target_tokens_per_batch = base_batch_size * target_seq_len

        for phase_str in schedule.split(','):
            parts = phase_str.strip().split(':')
            if len(parts) != 2:
                raise ValueError(f"Invalid phase format: {phase_str}")

            seq_len = int(parts[0])
            tokens_str = parts[1].strip().lower()

            # Parse token count
            if tokens_str == 'inf':
                end_tokens = float('inf')
            else:
                multipliers = {'k': 1_000, 'm': 1_000_000, 'b': 1_000_000_000}
                if tokens_str[-1] in multipliers:
                    end_tokens = int(float(tokens_str[:-1]) * multipliers[tokens_str[-1]])
                else:
                    end_tokens = int(tokens_str)

            # Calculate batch size to maintain constant tokens/batch
            batch_size = max(1, target_tokens_per_batch // seq_len)

            phases.append(ProgressivePhase(
                seq_len=seq_len,
                end_tokens=end_tokens,
                batch_size=batch_size,
            ))

        return cls(phases)

    def get_current_config(self, total_tokens: int) -> Tuple[int, int]:
        """
        Get current seq_len and batch_size for the given token count.

        Args:
            total_tokens: Total tokens seen so far

        Returns:
            (seq_len, batch_size) tuple
        """
        # Find current phase
        for i, phase in enumerate(self.phases):
            if total_tokens < phase.end_tokens:
                self.current_phase_idx = i
                return phase.seq_len, phase.batch_size

        # Past all phases, use the last one
        self.current_phase_idx = len(self.phases) - 1
        last_phase = self.phases[-1]
        return last_phase.seq_len, last_phase.batch_size

    def check_phase_transition(self, total_tokens: int) -> bool:
        """
        Check if we just transitioned to a new phase.

        Returns True only once per transition.
        """
        self.get_current_config(total_tokens)  # Updates current_phase_idx

        if self.current_phase_idx != self._last_phase_idx:
            self._last_phase_idx = self.current_phase_idx
            return True
        return False

    def get_phase_info(self) -> str:
        """Return human-readable phase information."""
        phase = self.phases[self.current_phase_idx]
        total_phases = len(self.phases)
        return f"Phase {self.current_phase_idx + 1}/{total_phases}: seq_len={phase.seq_len}, batch_size={phase.batch_size}"

    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            'current_phase_idx': self.current_phase_idx,
            '_last_phase_idx': self._last_phase_idx,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self.current_phase_idx = state_dict['current_phase_idx']
        self._last_phase_idx = state_dict['_last_phase_idx']
