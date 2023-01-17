import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)
    config.critic_hidden_dims = (256, 256)

    config.discount = 0.99

    config.expectile = 0.8  # The actual tau for expectiles.
    config.temperature = 3.0
    config.dropout_rate = None

    config.tau = 0.005  # For soft target updates.

    config.opt_decay_schedule = None  # Don't decay optimizer lr
    config.static_exploration_std = 0.03

    return config
