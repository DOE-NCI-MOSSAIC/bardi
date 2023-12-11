Examples
=============


Short Tutorial
**************************************************
.. code-block:: python

    import pandas as pd

    # create some sample data
    df = pd.DataFrame([
    {
        "patient_id_number": 1,
        "text": "The patient presented with notable changes in behavior, exhibiting increased aggression, impulsivity, and a distinct deviation from the Jedi Code. Preliminary examinations reveal a heightened midichlorian count and an unsettling connection to the dark side of the Force. Further analysis is warranted to explore the extent of exposure to Sith teachings. It is imperative to monitor the individual closely for any worsening symptoms and to engage in therapeutic interventions aimed at preventing further descent into the dark side. Follow-up assessments will be crucial in determining the efficacy of intervention strategies and the overall trajectory of the individual's alignment with the Force.",
        "dark_side_dx": "positive",
    },
    {
        "patient_id_number": 2,
        "text": "Patient exhibits no signs of succumbing to the dark side. Preliminary assessments indicate a stable midichlorian count and a continued commitment to Jedi teachings. No deviations from the Jedi Code or indicators of dark side influence were observed. Regular check-ins with the Jedi Council will ensure the sustained well-being and alignment of the individual within the Jedi Order.",
        "dark_side_dx": "negative",
    },
    {
        "patient_id_number": 3,
        "text": "The individual manifested heightened aggression, impulsivity, and a palpable deviation from established ethical codes. Initial examinations disclosed an elevated midichlorian count and an unmistakable connection to the dark side of the Force. Further investigation is imperative to ascertain the depth of exposure to Sith doctrines. Close monitoring is essential to track any exacerbation of symptoms, and therapeutic interventions are advised to forestall a deeper embrace of the dark side. Subsequent evaluations will be pivotal in gauging the effectiveness of interventions and the overall trajectory of the individual's allegiance to the Force.",
        "dark_side_dx": "positive",
    }