"""
Action gate — confirmation layer for irreversible real-world actions.

Any action that costs money, sends a message, places an order, or makes a
call requires explicit confirmation. The gate shows what will happen and
blocks until the user approves or denies.

This is not friction — it is the minimum boundary between intention and
consequence in the physical world.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass
class Action:
    kind:        str           # "call" | "sms" | "order" | "browse"
    description: str           # human-readable summary of what will happen
    cost:        float | None  # estimated cost in USD, None if unknown
    reversible:  bool          # can this be undone?
    payload:     dict          # the actual parameters


class ActionDenied(Exception):
    pass


def confirm(action: Action, auto_approve: bool = False) -> None:
    """
    Present the action to the user and block until approved or denied.
    Raises ActionDenied if the user rejects.
    """
    print(f"\n── Action Gate " + "─" * 50)
    print(f"  Kind:        {action.kind}")
    print(f"  Description: {action.description}")
    if action.cost is not None:
        print(f"  Cost:        ${action.cost:.2f}")
    print(f"  Reversible:  {'yes' if action.reversible else 'NO'}")
    print(f"  Payload:     {action.payload}")
    print("─" * 64)

    if auto_approve:
        print("  [auto-approved]\n")
        return

    try:
        answer = input("  Approve? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        raise ActionDenied("Gate interrupted")

    if answer not in ("y", "yes"):
        raise ActionDenied(f"Action denied by user: {action.kind}")
    print()
