import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import torch
    import matplotlib.pyplot as plt

    # Model: f(x) = sigmoid(k * (x - threshold))
    # k controls sharpness, threshold controls where the jump happens
    k = 50.0
    threshold = 0.8
    counter = 0
    def model(x):
        global counter
        counter += 1
        print(counter)
        sharp_feature = torch.sigmoid(k * (x[..., 0] - threshold))
        weak_features = 0.05 * x[..., 1] + 0.05 * x[..., 2]
        return sharp_feature + weak_features

    # Input: feature 0 is just past the threshold, others are arbitrary
    x  = torch.tensor([1.0, 0.5, 0.3]).unsqueeze(0) # input
    xb = torch.zeros(3).unsqueeze(0) # baseline

    # f(input) ≈ 1.0,  f(baseline) ≈ 0.0
    # So total attribution should sum to ≈ 1.0
    path_alphas = torch.linspace(0,1,10)
    def interpolate(alpha):
        return xb+alpha*(x-xb)
    path = torch.cat([interpolate(x) for x in path_alphas])
    preds = model(path)
    preds
    return model, plt, preds, x, xb


@app.cell
def _(plt, preds):
    plt.plot(preds)
    return


@app.cell
def _(attr_idg, model, plt, x, xb):
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(model)
    attr_igc, delta_igc = ig.attribute(
        inputs=x,
        baselines=xb,
        target=None,
        n_steps=30,
        return_convergence_delta=True,
        method="gausslegendre",
        internal_batch_size=1
    )
    attr_igc, delta_igc,plt.imshow(attr_idg,cmap="bwr")
    return


@app.cell
def _(model, plt, x, xb):
    from integrated_decision_gradients.integrated_gradients import get_integrated_gradients
    attr_ig, delta_ig = get_integrated_gradients(
        model=model,
        input=x,
        baseline=xb,
        target=None,
        n_steps=30,
        return_convergence_delta=True,
    )
    attr_ig,delta_ig,plt.imshow(attr_ig,cmap="bwr")
    return


@app.cell
def _(model, plt, x, xb):
    from integrated_decision_gradients import get_integrated_decision_gradients
    attr_idg, delta_idg = get_integrated_decision_gradients(
        model=model,
        input=x,
        baseline=xb,
        target=None,
        n_steps=30,
        return_convergence_delta=True,
        adaptive=False,
    )
    attr_idg,delta_idg,plt.imshow(attr_idg,cmap="bwr")
    return attr_idg, get_integrated_decision_gradients


@app.cell
def _(get_integrated_decision_gradients, model, plt, x, xb):
    attr_idga, delta_idga = get_integrated_decision_gradients(
        model=model,
        input=x,
        baseline=xb,
        target=None,
        n_steps=20,
        return_convergence_delta=True,
        adaptive=True,
    )
    attr_idga,delta_idga,plt.imshow(attr_idga,cmap="bwr")
    return


if __name__ == "__main__":
    app.run()
