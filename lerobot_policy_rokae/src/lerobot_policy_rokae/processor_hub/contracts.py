#!/usr/bin/env python

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContractSpec:
    """Canonical contract definition for Rokae data adaptation."""

    name: str
    observation_prefixes: tuple[str, ...] = field(default_factory=tuple)
    action_prefixes: tuple[str, ...] = field(default_factory=tuple)
    optional_prefixes: tuple[str, ...] = field(default_factory=tuple)


JOINT_CONTRACT = ContractSpec(
    name="joint",
    observation_prefixes=(
        "observation.joint_pos",
        "observation.left_joint_pos",
        "observation.right_joint_pos",
        "observation.gripper_pos",
        "observation.left_gripper_pos",
        "observation.right_gripper_pos",
    ),
    action_prefixes=(
        "joint_pos",
        "left_joint_pos",
        "right_joint_pos",
        "gripper_pos",
        "left_gripper_pos",
        "right_gripper_pos",
    ),
    optional_prefixes=("observation.images.",),
)

EE_CONTRACT = ContractSpec(
    name="ee",
    observation_prefixes=(
        "observation.cart_pos",
        "observation.left_cart_pos",
        "observation.right_cart_pos",
        "observation.gripper_pos",
        "observation.left_gripper_pos",
        "observation.right_gripper_pos",
    ),
    action_prefixes=(
        "cart_pos",
        "left_cart_pos",
        "right_cart_pos",
        "gripper_pos",
        "left_gripper_pos",
        "right_gripper_pos",
    ),
    optional_prefixes=("observation.images.",),
)

DUAL_CONTRACTS = {
    JOINT_CONTRACT.name: JOINT_CONTRACT,
    EE_CONTRACT.name: EE_CONTRACT,
}


def validate_contract_name(contract_name: str) -> None:
    if contract_name not in DUAL_CONTRACTS:
        raise ValueError(f"Unknown contract '{contract_name}'. Known contracts: {tuple(DUAL_CONTRACTS.keys())}")
