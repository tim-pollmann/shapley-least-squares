def check_number_of_samples_used(
    number_of_samples_used: int,
    target_number_of_samples,
    algorithm_name,
    max_deviation=0,
) -> None:
    if abs(number_of_samples_used - target_number_of_samples) > max_deviation:
        raise AssertionError(
            f'Algorithm "{algorithm_name}" exceeded the allowed deviation in sample usage.\n'
            f"Expected {target_number_of_samples} ± {max_deviation} samples, but got {number_of_samples_used} samples."
        )
