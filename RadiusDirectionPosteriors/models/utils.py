import math
LOG_2 = math.log(2.0)


def prior_summary(prior):
    direction_prior_type, direction_prior_param = prior['direction']
    concentration = direction_prior_param['concentration']
    direction_prior_str = '%s(%.2E)' % (direction_prior_type, concentration)
    radius_prior_type, radius_prior_param = prior['radius']
    radius_prior_str = radius_prior_type
    radius_prior_str += '(Local %.2E, Global %.2E)' % (radius_prior_param['tau_local'], radius_prior_param['tau_global'])
    return ', '.join([direction_prior_str, radius_prior_str])


def double_prior_summary(prior):
    direction_prior_param = prior['direction'][1]
    row_concentration = direction_prior_param['row_concentration']
    col_concentration = direction_prior_param['col_concentration']
    direction_prior_str = 'row - vMF(%.2E) col - vMF(%.2E)' % (row_concentration, col_concentration)
    radius_prior_type, radius_prior_param = prior['radius']
    radius_prior_str = radius_prior_type
    if radius_prior_type == 'Gamma':
        shape = radius_prior_param['shape']
        rate = radius_prior_param['rate']
        radius_prior_str += '(%.2E, %.2E)' % (shape, rate)
    elif radius_prior_type == 'Weibull':
        shape = radius_prior_param['shape']
        scale = radius_prior_param['scale']
        radius_prior_str += '(%.2E, %.2E)' % (shape, scale)
    elif 'HalfCauchy' in radius_prior_type:
        radius_prior_str += '(Local %.2E, Global %.2E)' % (radius_prior_param['tau_local'], radius_prior_param['tau_global'])
    else:
        raise NotImplementedError
    return ', '.join([direction_prior_str, radius_prior_str])