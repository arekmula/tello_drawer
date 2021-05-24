def check_limits(x: int, min_lim: int, max_lim: int) -> int:
    """Check that the value is within the limits

    Args:
        x (int): Variable to check
        min_lim (int): Minimum value of the variable
        max_lim (int): Maximum value of the variable

    Returns:
        int: Value between min_lim and max_lim
    """
    if x > max_lim:
        x = max_lim
    if x < min_lim:
        x = min_lim
    return x
