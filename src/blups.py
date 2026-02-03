import pandas as pd
import statsmodels.formula.api as smf

def compute_genotype_blups(pheno_long):

    df = pheno_long.copy()

    # Keep only the columns needed for BLUPs
    keep = ["value", "studyName", "germplasmName"]
    df = df[keep]

    # Drop rows missing required fields
    df = df.dropna(subset=keep)

    print("BLUP input shape after subsetting:", df.shape)
    print("BLUP input columns:", df.columns.tolist())

    # Mixed model: studyName fixed, genotype random
    formula = "value ~ C(studyName)"

    md = smf.mixedlm(
        formula,
        df,
        groups=df["germplasmName"],
    )
    mdf = md.fit(reml=True)

    # Extract genotype BLUPs
    re = mdf.random_effects
    blups = pd.DataFrame({
        "germplasmName": list(re.keys()),
        "blup": [v["Group"] for v in re.values()],
    })

    # Add intercept so BLUPs are on original scale
    mu = mdf.fe_params["Intercept"]
    blups["blup"] = blups["blup"] + mu

    return blups