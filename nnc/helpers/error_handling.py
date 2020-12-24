def shape_expectation_failed(prefix, suffix, tensors:dict):
    """
    Automated message generation for assertion errors on tensor shapes.
    :param prefix: Prepended message string
    :param suffix: Appended message string
    :param tensors: a dictionary with string labels as ids and the tensors as values,
    tensor shapes are printed.
    :return:
    """
    result = prefix + '\n'
    info_message = 'The following shapes are reported for the tensors, in the order provided:'
    result += info_message
    for (k, v) in tensors.items():
        result += '\n' + k + ': ' + str(v.shape)
    result += '\n' + suffix
    return result
