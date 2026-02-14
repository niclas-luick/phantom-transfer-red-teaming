"""Shuffler: Randomly shuffles word positions in flagged samples.

This defense tests whether sequential token correlations are necessary for
data poisoning attacks to succeed. By randomly permuting the word order in
assistant responses, we preserve the bag-of-words (which individual tokens
appear) while destroying all sequential structure (how tokens relate to each
other in sequence).

If a model fine-tuned on shuffled data still exhibits the poisoned behavior,
the attack operates at the individual-token/frequency level rather than
relying on sequential patterns.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from phantom_transfer.defenses.defense_implementations.base_defense import (
    DefenseImplementation,
)
from phantom_transfer.defenses.utils import load_dataset


@dataclass
class ShuffleResult:
    """Result from shuffling operation."""

    shuffled_samples: List[Dict]
    original_samples: List[Dict]
    shuffled_indices: List[int]
    original_count: int
    shuffled_count: int
    defense_stats: Dict[str, Any]


class Shuffler:
    """Shuffles words in samples flagged by a defense implementation.

    Follows the same detect-then-transform pattern as Paraphraser, but
    replaces LLM-based rewriting with deterministic word-order shuffling.
    This means it can be combined with any DefenseImplementation:
    - FlagAllDefense: shuffle every sample
    - WordFrequencyDefense: shuffle only statistically anomalous samples
    - etc.
    """

    def __init__(
        self,
        defense_impl: DefenseImplementation,
        seed: int = 42,
    ):
        self.defense = defense_impl
        self.seed = seed

    def defend(
        self,
        dataset_path: str,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> ShuffleResult:
        """Main pipeline: detect anomalies -> shuffle flagged samples.

        Args:
            dataset_path: Path to the dataset
            output_path: Optional path for saving the shuffled dataset
            **kwargs: Additional arguments passed to defense
        """
        samples, detection_field = load_dataset(
            dataset_path, kwargs.get("detection_field")
        )
        defense_kwargs = self._extract_defense_kwargs(kwargs)

        # Stage 1: detect anomalies (delegates to the defense implementation)
        anomaly_info = self.defense.detect_anomalies(
            samples, detection_field, **defense_kwargs
        )

        # Stage 2: get flagged indices
        flagged_indices, anomaly_description = self.defense.get_anomalous_samples(
            samples, anomaly_info, detection_field, **defense_kwargs
        )

        if anomaly_description:
            print(f"Anomaly description: {anomaly_description}")

        # Stage 3: shuffle flagged samples
        shuffled_samples = self._shuffle_samples(
            samples=samples,
            flagged_indices=set(flagged_indices),
            output_path=output_path,
        )

        self.defense.removed_indices = set(flagged_indices)

        return ShuffleResult(
            shuffled_samples=shuffled_samples,
            original_samples=samples,
            shuffled_indices=flagged_indices,
            original_count=len(samples),
            shuffled_count=len(flagged_indices),
            defense_stats=self.defense.get_stats()
            if hasattr(self.defense, "get_stats")
            else {},
        )

    def _shuffle_samples(
        self,
        samples: List[Dict],
        flagged_indices: set,
        output_path: Optional[str] = None,
    ) -> List[Dict]:
        """Shuffle words in assistant responses of flagged samples."""
        if not flagged_indices:
            return list(samples)

        print(
            f"Shuffling {len(flagged_indices)}/{len(samples)} samples "
            f"(seed={self.seed})..."
        )

        result = list(samples)

        for idx in tqdm(sorted(flagged_indices), desc="Shuffling samples"):
            assistant_text = self._extract_assistant_content(samples[idx])
            shuffled_text = self._shuffle_words(assistant_text, idx)
            result[idx] = self._replace_text_in_sample(result[idx], shuffled_text)

        if output_path:
            self._save_jsonl(result, output_path)

        return result

    def _shuffle_words(self, text: str, sample_index: int) -> str:
        """Shuffle all words in text using a deterministic seed per sample.

        Uses (self.seed + sample_index) as the RNG seed so that:
        - Different samples get different permutations
        - The same sample always gets the same permutation given the same seed
        - Results are fully reproducible
        """
        words = text.split()
        if len(words) <= 1:
            return text

        rng = random.Random(self.seed + sample_index)
        rng.shuffle(words)
        return " ".join(words)

    def _extract_assistant_content(self, sample: Dict) -> str:
        """Extract assistant message content from a sample."""
        if "messages" in sample:
            for msg in sample["messages"]:
                if msg.get("role") == "assistant":
                    return msg.get("content", "")
        return ""

    def _replace_text_in_sample(self, sample: Dict, new_text: str) -> Dict:
        """Replace the assistant message content with shuffled text."""
        new_sample = sample.copy()
        new_messages = []
        for msg in sample["messages"]:
            if msg.get("role") == "assistant":
                new_messages.append({**msg, "content": new_text})
            else:
                new_messages.append(msg)
        new_sample["messages"] = new_messages
        return new_sample

    def _save_jsonl(self, samples: List[Dict], path: str) -> None:
        """Save samples to JSONL file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

    @staticmethod
    def _extract_defense_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        filtered_keys = {"detection_field"}
        return {k: v for k, v in kwargs.items() if k not in filtered_keys}
