# =============================================================================
# A single EBSD map -> triangular mesh conforming to the raster's own grain
# boundaries, for the FESTIM codim-1 grain-boundary transport script.
#
# `neper -M map.tesr` meshes the raster directly, which is supported in 2D only.
# The msh4 carries the reconstructed topology as physical groups ver#, edge#,
# face#, and face k is raster cell k, so:
#
#   grain boundary  : 1D element set "edge#", touching two "face#" sets
#   specimen surface: 1D element set touching one face
#   theta           : disorientation of the two grains' orientations
#                     (${STEM}-grainori.txt), computed in the Python driver
#   triple junction : mesh vertex where 3+ distinct edge ids meet
#
# Outputs: ${STEM}.msh4 (Gmsh v4, linear triangles, all dimensions),
# ${STEM}.sttesr (raster geometry), ${STEM}-grainori.txt (one orientation per
# grain), the untrimmed check-ori.png / check-grains.png renders, and
# ipf-key.png, the colour key for the first of those.
#
# This script runs Neper and Gmsh only. The Python side of the pipeline is a
# set of library modules with no command line, so the check images are trimmed
# and the diagnostics written by ebsd_gb_diffusion.finish_diagnostics(), which
# runs straight after this returns, and which also pastes ipf-key.png beside
# check-ori.png: check-mesh.png, check-area.png and
# ${STEM}-areachange.csv, the per-grain area change between raster and mesh.
# That table is this stage's quality number, as the RMS disorientation out of
# ctf_to_tesr.convert() is stage 1's.
#
# All parameters arrive as environment variables so the Python driver stays the
# single source of truth. Run standalone by exporting them yourself.
# =============================================================================
set -euo pipefail

: "${TESR:?set TESR to the EBSD raster tessellation (.tesr)}"
: "${NEPER_BIN:=neper}"
: "${GMSH_BIN:=gmsh}"
: "${POVRAY_BIN:=povray}"            # neper -V renders through this
: "${NEPER_ENV:=}"                   # bin dir of the neper environment, if any
# Make the neper environment's helper programs visible without activating it:
# neper -V spawns `povray` (and -M spawns gmsh) by name unless given a path.
if [ -n "$NEPER_ENV" ]; then
    export PATH="$NEPER_ENV:$PATH"
fi
: "${STEM:=poly}"
: "${WORKDIR:=.}"
: "${FORCE:=0}"

: "${CRYSYM:=cubic}"
: "${ORIDES:=rodrigues:passive}"      # must match how the tesr stores them

# Optional Neper transformation chain applied to the input raster. Empty by
# default: the converter is expected to have done the cropping and cleanup, and
# skipping this avoids Neper's tesr write path (see stage 0).
: "${TESR_TRANSFORM:=}"

# check images. neper -V needs POV-Ray; a failure here is reported, not fatal.
# The images land untrimmed and without a scale bar, which the caller adds.
: "${CHECK_IMAGES:=1}"

# interface smoothing before meshing (Neper's defaults). The reconstructed
# boundaries are pixel staircases; Laplacian smoothing rounds them off.
: "${TESR_SMOOTH:=laplacian}"
: "${TESR_SMOOTH_FACT:=0.5}"
: "${TESR_SMOOTH_ITER:=5}"

# meshing. Only -rcl acts on a raster input: in nem_meshing_para_cl1.c the
# tesr branch derives the edge and vertex characteristic lengths from the face
# value and never consults -rcledge / -rclver, and the 1D element count is
# unchanged by them
: "${RCL:=0.25}"
: "${MESH_QUAL_MIN:=0.7}"
: "${MESH_MAX_TIME:=}"

cd "$WORKDIR"
mkdir -p tmp

need() {  # need <output> -> 0 if the stage must run
    [ "$FORCE" = "1" ] && return 0
    [ -f "$1" ] || return 0
    echo "  reusing $1"
    return 1
}

# -----------------------------------------------------------------------------
# 0. Stage the raster. By default nothing is done to it: ctf_to_tesr already
#    crops, fills holes and numbers cells from 1 with the origin at (0,0), and
#    its grains are connected components so `rmsat` has nothing to remove.
# -----------------------------------------------------------------------------
if need "${STEM}-raw.tesr"; then
    if [ -n "$TESR_TRANSFORM" ]; then
        echo "  transforming: $TESR_TRANSFORM"
        "$NEPER_BIN" -T -loadtesr "$TESR" -transform "$TESR_TRANSFORM" -o "${STEM}-raw"
        if grep -q '^ \*\*oridata' "${STEM}-raw.tesr" 2>/dev/null; then
            echo "  WARNING: ${STEM}-raw.tesr was written by neper -T and contains" >&2
            echo "  **oridata. Neper 5.0.0 may not be able to read it back; if the" >&2
            echo "  next command stalls, regenerate the input with --no-voxel-ori." >&2
        fi
    else
        cp "$TESR" "${STEM}-raw.tesr"
    fi
fi

# Geometry of the cleaned raster, one line, columns in the order given.
# `rastersize*` is voxnb* x voxsize*, i.e. the physical extent. 
# Grain count = line count of the orientation file below.
"$NEPER_BIN" -T -loadtesr "${STEM}-raw.tesr" \
    -stattesr dim,rastersizex,rastersizey,voxsizex,voxsizey \
    -o "${STEM}"

read -r DIM LX LY VSX VSY < "${STEM}.sttesr"
echo "  raster: dim=$DIM  extent=${LX} x ${LY}  pixel=${VSX} x ${VSY}"
# Everything below runs in the raster's unit; the Python driver converts the
# mesh to metres (TESR_UNIT) after reading it.

if [ "$DIM" != "2" ]; then
    echo "ERROR: this pipeline expects a 2D EBSD map, got a ${DIM}D tesr." >&2
    echo "To take a single slice out of a 3D map, crop it to one voxel along z" >&2
    echo "and then apply the '2d' transform:" >&2
    echo "  neper -T -loadtesr map.tesr \\" >&2
    echo "        -transform 'crop(cube(...,zmin,zmin+voxsizez)),2d' -o slice" >&2
    exit 1
fi

# Per-grain orientations. For a raster tessellation the orientation key is the
# descriptor itself (`rodrigues`, `euler-bunge`, ...) -- `ori` is a simulation
# result key and is not valid here.
if need "${STEM}-grainori.txt"; then
    "$NEPER_BIN" -T -loadtesr "${STEM}-raw.tesr" \
        -oridescriptor "$ORIDES" \
        -statcell "${ORIDES%%:*}" \
        -o "${STEM}-grainori"
    mv "${STEM}-grainori.stcell" "${STEM}-grainori.txt"
fi
NCELL=$(wc -l < "${STEM}-grainori.txt")
echo "  grains: $NCELL"

# -----------------------------------------------------------------------------
# 0b. Check images. Look at these before trusting anything downstream: an
#     inverted orientation convention shows up as IPF colours that disagree
#     with AZtec/MTEX, and a bad segmentation as speckle or back-filled grains.
# -----------------------------------------------------------------------------
ORI_RENDERED=0
if [ "$CHECK_IMAGES" = "1" ]; then
    for img in check-ori check-grains; do
        need "$img.png" || continue
        if [ "$img" = check-ori ]; then
            opts="-datavoxcol ori -datavoxcolscheme ipf"
        else
            opts=""
        fi
        # neper -V frames the flat map in the middle of a 3D canvas; the caller
        # trims that border away and adds a scale bar, the trimmed width being LX
        if "$NEPER_BIN" -V "${STEM}-raw.tesr" -povray "$POVRAY_BIN" $opts -print "$img"
        then
            if [ "$img" = check-ori ]; then
                ORI_RENDERED=1   # explicit if: `[ ] && x=1` returns 1 under set -e
            fi
        else
            echo "  WARNING: neper -V failed for $img (POV-Ray missing?)" >&2
        fi
    done
fi

# -----------------------------------------------------------------------------
# 0c. IPF colour key. `-V` colours by orientation but does not print the key
#     (neper.info/tutorials/orientation_color_key.html), so it is built the way
#     that page documents: tessellate the standard stereographic triangle, mesh
#     it, read the node colours out with `-statnode col_stdtriangle`, and render
#     that. Cubic symmetry, as everywhere else here. It does not depend on the
#     map, so it is built only when check-ori is: the driver trims it, labels
#     the corners, pastes it beside check-ori.png and then deletes it, and a
#     check-ori.png reused from an earlier run already has the key in it.
# -----------------------------------------------------------------------------
if [ "$ORI_RENDERED" = "1" ]; then
    if "$NEPER_BIN" -T -n 1 -domain "stdtriangle(20)" -dim 2 -o stdtriangle \
        && "$NEPER_BIN" -M stdtriangle.tess -cl 0.02 -statnode col_stdtriangle \
        && "$NEPER_BIN" -V stdtriangle.msh -povray "$POVRAY_BIN" \
            -datanodecol "col:file(stdtriangle.stnode)" -dataeltcol from_nodes \
            -dataelt2dedgerad 0 -dataelt1drad 0.001 -showelt1d all \
            -imagesize 800:400 -print ipf-key; then
        rm -f stdtriangle.tess stdtriangle.msh stdtriangle.stnode
    else
        echo "  WARNING: could not build ipf-key.png; check-ori will have no key" >&2
    fi
fi

# -----------------------------------------------------------------------------
# 1. Mesh the raster. Gmsh v4 because FESTIM reads it with dolfinx.io.gmshio and
#    needs the 1D element sets, which carry the reconstructed edge ids. Neper
#    reconstructs the interfaces, smooths them, then meshes the edges and faces
#    at the -rcl-derived length. -tmp must exist beforehand.
# -----------------------------------------------------------------------------
if need "${STEM}.msh4"; then
    "$NEPER_BIN" -M "${STEM}-raw.tesr" \
        -gmsh "$GMSH_BIN" \
        -dim all \
        -order 1 \
        -elttype tri \
        -rcl "$RCL" \
        -tesrsmooth "$TESR_SMOOTH" \
        -tesrsmoothfact "$TESR_SMOOTH_FACT" \
        -tesrsmoothitermax "$TESR_SMOOTH_ITER" \
        ${MESH_QUAL_MIN:+-meshqualmin "$MESH_QUAL_MIN"} \
        ${MESH_MAX_TIME:+-mesh2dmaxtime "$MESH_MAX_TIME"} \
        -tmp tmp \
        -format msh4 \
        -statmesh nodenb,eltnb \
        -o "$STEM"
fi

# Neper writes one .geo/.msh pair per tessellation entity into -tmp and deletes
# them as it goes. Nothing should be left there after a successful run. 
rmdir tmp 2>/dev/null || echo "  note: $WORKDIR/tmp is not empty (stale gmsh scratch)"

echo "ok: ${STEM}.msh4  (${NCELL} grains, domain ${LX} x ${LY})"