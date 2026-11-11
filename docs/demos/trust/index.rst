:hide-toc:
:og:description: Replay the XAI 2024 trust case study step by step: VAE-GAN reconstruction, threshold, LIME and SHAP explanations, and the optimal Jaccard check.

Inside the Trust Case Study: Detect, Decide, Explain, Verify
============================================================

Can an anomaly detector be right for the wrong reason? This demo replays the
pipeline of the paper, one stage at a time, on the eight test images discussed
in its Fig. 3 and Fig. 4: how a VAE-GAN trained only on good parts scores a
hazelnut or a screw, where one threshold splits the test set, how LIME and SHAP
attribute the reconstruction error to regions of the image, and whether those
regions overlap the annotated defect.

.. raw:: html

   <div class="adx-frame">
     <iframe id="adx-demo" src="app/index.html" scrolling="no"
             title="Interactive demo of the trust case study"></iframe>
   </div>
   <p class="adx-open"><a href="app/index.html">Open the demo on its own page</a></p>
   <script>
   (function () {
     var frame = document.getElementById("adx-demo");
     function doc() { try { return frame.contentDocument; } catch (e) { return null; } }
     // Furo stores the reader's choice on <body data-theme="light|dark|auto">.
     function syncTheme() {
       var d = doc(); if (!d) return;
       var t = document.body.dataset.theme, root = d.documentElement;
       if (t === "light" || t === "dark") root.setAttribute("data-theme", t);
       else root.removeAttribute("data-theme");
     }
     function fit() {
       var d = doc(); if (!d) return;
       frame.style.height = d.documentElement.scrollHeight + "px";
     }
     frame.addEventListener("load", function () {
       syncTheme(); fit();
       var d = doc();
       if (d && window.ResizeObserver) new ResizeObserver(fit).observe(d.body);
     });
     new MutationObserver(syncTheme).observe(document.body, { attributes: true, attributeFilter: ["data-theme"] });
   })();
   </script>

Each stage is linked to the sample chosen at the top: point at an image to read
pixel values, drag the threshold, step through LIME's perturbations, and move
the cut θ that turns an explanation into a mask.

How it was computed
-------------------

Everything on the page comes from the paper's own pipeline, run unchanged with
the trained 30 000-epoch checkpoints in ``models/``:

.. code-block:: python

   def f(batch):                                   # the paper's lime_predict_loss
       mu, _, _ = model.encoder(batch)
       return ((model.decoder(mu) - batch) ** 2).sum(axis=(1, 2, 3))[:, None]

   segments = search_segment_number((xi + recon) / 2, 100)          # S1
   lime = LimeImageExplainer(random_state=1234).explain_instance(
       xi, f, hide_color=recon, num_samples=5000, segmentation_fn=lambda _: segments)
   shap_values = shap.Explainer(f, shap.maskers.Image(recon, recon.shape))(
       xi[None], max_evals=5000)

The anomaly map is :math:`m = \max_c |\xi' - \xi|` and the anomaly score is
:math:`\alpha = \max(m)`. Setting S2 segments
:math:`0.33\,\xi + 0.33\,\xi' + 0.34\,\gamma` instead, so that the superpixels
follow the defect. The Jaccard curves use every distinct value of the
explanation as a cut, as ``calc_IoU_curve`` does in the paper's ``utils.py``.

The compute script wraps LIME's ``data_labels`` to log every perturbation it
draws and every loss the model returns. What the page shows after *n*
perturbations is LIME's own ridge regression refitted on the first *n* rows of
that log, and the script stops with an error unless the fit on all rows equals
the explanation ``explain_instance`` returned.

Stage 2 covers the full test sets: anomaly scores and Jaccard maxima for every
image come from the result CSVs archived on the ``xai-2024-paper`` branch, with
the score axis read back from the vector PDFs of the paper's Fig. 2.

.. note::

   The published data was built in preview mode, without the MVTec AD files:
   each 128 px input and ground-truth mask was recovered from the SVG figures
   archived on ``xai-2024-paper`` by inverting matplotlib's resampling. Anomaly
   scores agree with the paper's to within 0.007 and SHAP's best Jaccard index
   matches on six of the eight samples, while LIME, which depends on the
   superpixels, can differ. A rebuild from the dataset replays the paper exactly.

Run it again
------------

The scripts are in ``docs/demos/trust/compute``. ``config.toml`` lists the
samples, the budgets and the dataset path; change it to explain other images.
TensorFlow 2.15 needs Python 3.9–3.11.

.. code-block:: bash

   pip install -r requirements.txt pymupdf scipy
   git fetch origin xai-2024-paper            # archived results used by pack.py
   cd docs/demos/trust/compute
   python explain.py --mvtec-root /path/to/MVTec   # about 3 minutes per sample on a CPU
   python pack.py                                  # the demo's data, ../app/assets/data.js
