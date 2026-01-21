from enum import Enum, auto

class Decision(Enum):
    IGNORE = auto()
    MONITOR = auto()
    INTRUSION_CONFIRMED = auto()

class DecisionEngine:
    """
    Central decision logic.
    NO side effects (no alarm, no IO) — only decisions.
    """

    def decide(
        self,
        label: str,
        category: str,
        inside_roi: bool,
        face_identity: dict | None = None
    ) -> Decision:

        # 1️⃣ Must be inside ROI
        if not inside_roi:
            return Decision.IGNORE

        # 2️⃣ Only persons trigger higher logic
        if category != "intrusion":
            return Decision.MONITOR

        # 3️⃣ If face recognition available
        if face_identity is not None:
            if face_identity.get("known", False):
                # Known person → monitor only
                return Decision.MONITOR
            else:
                # Unknown person → confirmed intrusion
                return Decision.INTRUSION_CONFIRMED

        # 4️⃣ Person inside ROI, no face info yet
        return Decision.INTRUSION_CONFIRMED
