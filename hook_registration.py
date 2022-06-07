# torch
import pytorch_lightning as pl

# project
import ckconv
import models
import torch

# typing
from omegaconf import OmegaConf

from functools import partial


class HookTriggerCallback(pl.callbacks.Callback):
    def __init__(self, registered_hook, triggers, timeout=1, log_output=False):
        """We want to control how often a hook is triggered through pl callbacks. This class is implemented as a pl
        callback, where each callback function specified in 'triggers' serves as a one-time trigger for the forward
        hook.

        :param registered_hook: The forward hook, a function of the form;
            ''' def forward_hook(module: torch.nn.Module, input: torch.tensor, output: torch.tensor, name: str):
                    ...
            '''

        :param triggers: List of pl callback functions to serve as one-time trigger, e.g.
            ['on_train_batch_start', 'on_epoch_end']

        :param timeout: If this value is not 1 but for example 4, only trigger every 4th step.

        :param log_output: Whether or not to log the hook output locally. @todo implement this.
        """
        self.registered_hook = registered_hook
        self.opened = False
        self.handle = None

        # We may only want to open the hook once every n times the trigger has been activated.
        assert type(timeout) == int and timeout >= 1
        self.timeout = timeout

        # register hook triggers, these are pl callback functions e.g. 'on_train_batch_start',
        # we overwrite these functions with a lambda function opening the hook.
        for trigger in triggers:
            setattr(self, trigger, lambda *args: self.open(trigger))
            setattr(self, trigger + "_counter", 0)

    def open(self, trigger):
        """Check whether we need to open the hook for this trigger.

        :param trigger: Function name that served as trigger for this function call.
        """

        # Get count of how many times this trigger has been activated.
        trigger_counter = getattr(self, trigger + "_counter")

        # Check whether we want the hook to open based on the timeout of this hook.
        if not trigger_counter % self.timeout:
            self.opened = True

        # Increment the counter.
        setattr(self, trigger + "_counter", trigger_counter + 1)

    def close(self):
        self.opened = False

    def remove_hook(self):
        self.handle.remove()

    def add_handle(self, handle):
        self.handle = handle

    def __call__(self, module, input, output):
        """Each forward pass through a module that this hook has been registered to will result in a call to this
        function. We only want the hook to actually execute if a trigger (pl callback function) has been executed.
        """
        if self.opened:
            self.close()

            # Make sure we don't track gradients for the hook functions.
            with torch.no_grad():
                self.registered_hook(module, input, output)


def register_hooks(
    cfg: OmegaConf,
    model: pl.LightningModule,
):
    """Register hooks to a given model

    :param cfg: config file
    :param model: model to inject hooks into
    :param hook: function to hook onto modules,
        gets args as follows: func(mod, in, out, mod_name)
    :param hook_onto: which modules to hook onto

        For example, to hook a visualisation function onto the output of all kernel_nets:

            ...
            register_hooks(model, visualize_kernel_out_hook, [ckconv.ck.nn.SIRENBase, ckconv.ck.nn.MFNBase])
            ...
    """

    # pl reattaches callbacks at every .validate() and .train() call, and we don't want doubly registered hooks
    # this removes any preexisting hooks
    if model.trainer:
        for callback in model.trainer.callbacks:
            if isinstance(callback, HookTriggerCallback):
                callback.remove_hook()

    callbacks = []
    for hook_cfg in cfg.hooks:

        hook_fn = hook_cfg.function
        hook_onto_mods = hook_cfg.hook_onto
        hook_triggers = hook_cfg.triggers
        hook_limits = hook_cfg.limit_to
        hook_timeout = hook_cfg.timeout

        # get the hook function
        hook = getattr(ckconv.utils.hooks, hook_fn)

        # get the module type to hook onto
        hook_onto = []
        for mod in hook_onto_mods:

            # very hacky was to obtain the class we want the hook to latch onto
            module_name = ".".join(mod.split(".")[:-1])
            class_name = mod.split(".")[-1]
            hook_onto.append(getattr(eval(module_name), class_name))

        # except if we want to look at the last module
        if hook_limits == "last":
            named_modules = reversed(list(model.named_modules()))
        # iterate forwards over modules
        else:
            named_modules = model.named_modules()

        for name, module in named_modules:

            # isinstance also recognises subclasses
            if any(isinstance(module, m) for m in hook_onto):

                # print hook info
                print(f"Registering {hook_cfg.type} hook '{hook.__name__}' to '{name}'")

                # for every registered forward hook, we create a corresponding pl callback class.
                one_time_hook = HookTriggerCallback(
                    registered_hook=partial(hook, name=name),
                    triggers=hook_triggers,
                    timeout=hook_timeout,
                )

                # register the hook
                if hook_cfg.type == "forward":
                    h = module.register_forward_hook(one_time_hook)
                elif hook_cfg.type == "backward":
                    h = module.register_backward_hook(one_time_hook)

                # save the handle
                one_time_hook.add_handle(h)

                # append to list of callbacks that will be merged with trainer callbacks
                callbacks.append(one_time_hook)

                if hook_limits == "first" or hook_limits == "last":
                    break

    return callbacks
