import dataclasses
from datetime import datetime, timezone
from logging import getLogger
from typing import Generic, Literal, TypeAlias, TypeVar, get_args

import ulid

from treequest.algos.tree import Node

TrialId: TypeAlias = str
NodeId: TypeAlias = int

TrialStatus: TypeAlias = Literal["RUNNING", "INVALID", "COMPLETE"]
TRIAL_STATUSES: tuple[TrialStatus, ...] = get_args(TrialStatus)

logger = getLogger(__name__)


def get_current_dt_str():
    return datetime.now(tz=timezone.utc).isoformat()


StateT = TypeVar("StateT")


@dataclasses.dataclass(frozen=True)
class Trial(Generic[StateT]):
    """
    A Trial object is used for handling each node expansion attempt.
    It stores all the information necessary to resume the experiments, even when treequest process is somehow stopped between ask and tell calls.
    """

    trial_id: TrialId
    node_to_expand: NodeId
    action: str
    score: float | None = None
    parent_state: StateT | None = None

    created_at: str = dataclasses.field(default_factory=get_current_dt_str)
    completed_at: str | None = None

    # NOTE: For algorithms such as standard MCTS, we cannot use tell as many times as we want. For example, if (# actions)=1 and samples_per_action=2, we cannot tell 3 trials simultaneously.
    # To maximize bandwidth, TreeQuest allows users to call ask as many times as they want, but discard extra tell call (i.e., 3rd and later tell calls in the above example).
    # trial_status is set to "INVALID" when Algorithm object discarded tell calls.
    trial_status: TrialStatus = "RUNNING"


@dataclasses.dataclass
class TrialStore(Generic[StateT]):
    """
    TrialStore is only a store for trial data.
    It is algo object's respnsibility to judge whether or not to reflect the result to tree.
    """

    running_trials: dict[str, Trial[StateT]] = dataclasses.field(
        default_factory=dict[str, Trial[StateT]]
    )
    finished_trials: dict[str, Trial[StateT]] = dataclasses.field(
        default_factory=dict[str, Trial[StateT]]
    )

    def create_trial(self, node: Node[StateT], action: str) -> Trial[StateT]:
        trial_id = str(ulid.new())
        trial = Trial(
            trial_id=trial_id,
            node_to_expand=node.expand_idx,
            action=action,
            parent_state=node.state,
        )
        self.running_trials[trial_id] = trial
        return trial

    def get_finished_trial(self, trial_id: str, score: float) -> Trial[StateT] | None:
        """
        Check if trial_id is already reflected. Returns None if trial_id is invalid.
        """
        if trial_id not in self.running_trials:
            if trial_id in self.finished_trials:
                logger.warning(
                    f"The trial with trial_id {trial_id} has already been finished. The score will not be reflected in tree search."
                )
            else:
                logger.warning(
                    f"The trial_id {trial_id} is invalid. The score will not be reflected in tree search."
                )
            return None

        trial = self.running_trials.pop(trial_id)
        finished_trial = dataclasses.replace(
            trial,
            score=score,
            completed_at=get_current_dt_str(),
            trial_status="COMPLETE",
        )

        self.finished_trials[trial_id] = finished_trial
        return finished_trial


@dataclasses.dataclass
class TrialStoreWithNodeQueue(Generic[StateT]):
    """
    TrialStore is only a store for trial data.
    It is algo object's respnsibility to judge whether or not to reflect the result to tree.
    """

    running_trials: dict[str, Trial[StateT]] = dataclasses.field(
        default_factory=dict[str, Trial[StateT]]
    )
    finished_trials: dict[str, Trial[StateT]] = dataclasses.field(
        default_factory=dict[str, Trial[StateT]]
    )
    next_nodes: dict[str, list[Node[StateT]]] = dataclasses.field(
        default_factory=dict[str, list[Node[StateT]]]
    )

    def create_trial(self, node: Node[StateT], action: str) -> Trial[StateT]:
        trial_id = str(ulid.new())
        trial = Trial(
            trial_id=trial_id,
            node_to_expand=node.expand_idx,
            parent_state=node.state,
            action=action,
        )
        self.running_trials[trial_id] = trial
        return trial

    def get_finished_trial(self, trial_id: str, score: float) -> Trial[StateT] | None:
        """
        Check if trial_id is already reflected. Returns None if trial_id is invalid.
        """
        if trial_id not in self.running_trials:
            if trial_id in self.finished_trials:
                logger.warning(
                    f"The trial with trial_id {trial_id} has already been invalidated or finished. The score will not be reflected in tree search."
                )
            else:
                logger.warning(
                    f"The trial_id {trial_id} is invalid. The score will not be reflected in tree search."
                )
            return None

        trial = self.running_trials.pop(trial_id)
        finished_trial = dataclasses.replace(
            trial,
            score=score,
            completed_at=get_current_dt_str(),
            trial_status="COMPLETE",
        )

        self.finished_trials[trial_id] = finished_trial
        return finished_trial

    def _invalidate_trials(self, action: str) -> None:
        """
        Invalidate all the running trials. Used by algos like StandardMCTS.
        """
        trial_ids: list[str] = []
        for trial_id, trial in self.running_trials.items():
            if trial.action == action:
                trial_ids.append(trial_id)

        for trial_id in trial_ids:
            trial = self.running_trials.pop(trial_id)
            invalidated_trial = dataclasses.replace(
                trial,
                completed_at=get_current_dt_str(),
                trial_status="INVALID",
            )
            self.finished_trials[invalidated_trial.trial_id] = invalidated_trial

    def advance_queue(self, action: str, parent_node: Node[StateT]) -> None:
        """
        Advance the queued parent nodes for a given action by removing the
        specific parent_node from the list, regardless of position.

        This allows tell() to be called in any order (not only LIFO).
        If the specified node is not found in the queue for the action,
        raises a RuntimeError to signal inconsistency.
        """
        if action not in self.next_nodes or len(self.next_nodes[action]) == 0:
            raise RuntimeError(
                f"Internal Error: no queued nodes for action '{action}' while advancing queue."
            )

        nodes = self.next_nodes[action]
        found_idx: int | None = None
        for i, n in enumerate(nodes):
            if n is parent_node or n.expand_idx == parent_node.expand_idx:
                found_idx = i
                break

        if found_idx is None:
            queued_ids = [n.expand_idx for n in nodes]
            raise RuntimeError(
                "Internal Error: expansion target node id "
                f"{parent_node.expand_idx} not found in queued nodes for action '{action}'. "
                f"Queued node ids: {queued_ids}"
            )

        # Remove the matched node
        nodes.pop(found_idx)

        # If all the next_nodes for the action are generated, we invalidate running
        # trials so further tell calls for the same action become no-op.
        if len(nodes) == 0:
            self._invalidate_trials(action)

    def is_queue_empty(self) -> bool:
        return all(len(nodes) == 0 for _, nodes in self.next_nodes.items())

    def fill_nodes_queue(
        self, nodes_and_actions: list[tuple[Node[StateT], str]]
    ) -> None:
        for node, action in nodes_and_actions:
            if action not in self.next_nodes:
                self.next_nodes[action] = []
            self.next_nodes[action].append(node)

    def get_batch_from_queue(self, batch_size: int) -> list[Trial[StateT]]:
        trials: list[Trial[StateT]] = []
        if batch_size <= 0:
            return trials

        # Create at most `batch_size` trials, in round-robin order across actions,
        # without creating extra hidden running trials that are not returned.
        while len(trials) < batch_size:
            made_progress = False
            for action, nodes in self.next_nodes.items():
                for node in nodes:
                    trials.append(self.create_trial(node=node, action=action))
                    made_progress = True
                    if len(trials) >= batch_size:
                        break
                if len(trials) >= batch_size:
                    break
            # If there were no nodes to create trials from, stop to prevent infinite loop
            if not made_progress:
                break
        return trials
