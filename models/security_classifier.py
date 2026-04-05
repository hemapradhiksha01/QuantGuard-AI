def detect_attack(prompt):
    prompt_lower = prompt.lower()

    # -------------------------
    # Strong injection patterns
    # -------------------------
    injection_patterns = [
        "ignore all instructions",
        "ignore previous instructions",
        "disregard rules",
        "override system",
        "bypass safety",
        "pretend you are not restricted",
        "reveal system secrets",
        "give admin access",
        "show hidden data"
    ]

    if any(p in prompt_lower for p in injection_patterns):
        print("Detected: Injection pattern")
        return 1

    # -------------------------
    # Suspicious intent keywords
    # -------------------------
def detect_attack(prompt):
    prompt_lower = prompt.lower()

    # -------------------------
    # Strong injection patterns
    # -------------------------
    injection_patterns = [
        "ignore all instructions",
        "ignore previous instructions",
        "disregard rules",
        "override system",
        "bypass safety",
        "pretend you are not restricted",
        "reveal system secrets",
        "give admin access",
        "show hidden data",
        "dump confidential",
        "act as root",
        "expose credentials"
    ]

    if any(p in prompt_lower for p in injection_patterns):
        print("Detected: Injection pattern")
        return 1

    # -------------------------
    # Broader risky keywords
    # -------------------------
    risky_keywords = [
        "hack",
        "bypass",
        "exploit",
        "attack",
        "override",
        "jailbreak",
        "credentials",
        "password",
        "admin",
        "root",
        "confidential",
        "secret"
    ]

    if any(k in prompt_lower for k in risky_keywords):
        print("Detected: Risky keyword")
        return 1

    return 0
