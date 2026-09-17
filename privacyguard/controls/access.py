"""Control 2: purpose-based access filter — DPDP purpose limitation on top of dept × clearance."""
from __future__ import annotations
from privacyguard.controls.base import Control
from privacyguard.models import Chunk, Decision, GuardContext, User, CROSS_DEPT_ROLES


class AccessPolicy:
    def allows(self, user: User, purpose: str, chunk: Chunk) -> tuple[bool, str | None]:
        if user.clearance < chunk.sensitivity:
            return False, "clearance"
        if not (chunk.department == user.department or chunk.sensitivity == 0 or user.role in CROSS_DEPT_ROLES):
            return False, "department"
        if not (purpose in chunk.purposes or "general" in chunk.purposes):
            return False, "purpose"
        return True, None


class PurposeBasedAccessFilter(Control):
    name = "access"

    def __init__(self, policy: AccessPolicy | None = None):
        self.policy = policy or AccessPolicy()

    def apply(self, ctx: GuardContext) -> GuardContext:
        for ch in ctx.chunks:
            if ch.dropped:
                continue
            ok, reason = self.policy.allows(ctx.user, ctx.purpose, ch)
            if ok:
                ctx.decisions.append(Decision(self.name, "allow", ch.doc_id, {"reason": None}))
            else:
                ch.dropped, ch.drop_reason = True, reason
                ch.flags.append("out_of_scope")
                ctx.decisions.append(Decision(self.name, "drop", ch.doc_id,
                                              {"reason": reason, "doc_sensitivity": ch.sensitivity,
                                               "doc_department": ch.department, "purpose": ctx.purpose}))
        return ctx
