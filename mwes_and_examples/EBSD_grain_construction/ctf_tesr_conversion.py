from ctf_to_tesr import convert

res = convert(
    "mwes_and_examples/EBSD_grain_construction/D7 PBF SS316L.ctf",
    "mwes_and_examples/EBSD_grain_construction/d7.tesr",
    min_pixels=15,
    diagnostics=True,
    max_mad=1.5,
    allow_error=True,
    crop="0,306,0,306",
)
res["segmentation_error"]["indexed"]["rms"]  # degrees
