"""Maximum-likelihood mass fits to LHCb Upsilon dimuon data (DAH Project F1).

Typical use::

    import upsilon as ups

    events = ups.load("small")
    model = ups.build_model(("1S",), window=(9.0, 9.9))
    result = ups.fit(model, events["mass"])
    print(result.summary())
    ups.plotting.plot_fit(result, save="fit_1S.png")
"""
from . import models, plotting
from .fitting import FitResult, fit
from .io import COLUMNS, load, load_events, select
from .models import MassModel, build_model

__all__ = [
    "COLUMNS", "FitResult", "MassModel", "build_model", "fit", "load", "load_events",
    "models", "plotting", "select",
]
