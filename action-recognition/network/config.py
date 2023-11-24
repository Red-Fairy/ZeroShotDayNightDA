import logging

def get_config(name, is_dark, **kwargs):

	logging.debug("loading network configs of: {}".format(name.upper()))

	config = {}

	if not is_dark:
		logging.info("Preprocessing:: using default mean & std from Kinetics original.")
		config['mean'] = [0.43216, 0.394666, 0.37645] # Setting for non ARID
		config['std'] = [0.22803, 0.22145, 0.216989] # Setting for non ARID

	else:
		logging.info("Preprocessing:: using default mean & std from dark dataset.")
		config['mean'] = [0.079612, 0.073888, 0.072454] # Setting for ARID
		config['std'] = [0.100459, 0.0970497, 0.089911] # Setting for ARID

	logging.info("data:: {}".format(config))
	return config
