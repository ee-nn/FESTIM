# =============================================================================
# A single EBSD map -> triangular mesh conforming to the raster's own grain
# boundaries, for the FESTIM codim-1 grain-boundary transport script.
#
# This is "route A": `neper -M map.tesr` meshes the raster directly, which
# Neper supports in 2D only (neper_m.html: "Free meshing of raster
# tessellations works for 2D tessellations only"). Neper reconstructs the
# interfaces of the raster into a vertex/edge/face topology, smooths them
# (-tesrsmooth) and meshes that, so the mesh conforms to the measured
# boundaries, non-convex ones included. Route B (fitting a convex-cell Laguerre
# tessellation with -T -morpho tesr) was abandoned because its objective has a
# floor set by grain convexity: with a median grain solidity of ~0.8 it left
# ~17 % of the voxels in the wrong cell however long it ran.
#
# What route A does not give is a .tess, so there is no -statedge / -statver:
# `-format tess` on a raster input segfaults in Neper 5.0.0 while "Writing
# geometry results". Everything the transport script needs is recovered from
# the mesh instead. The msh4 carries the reconstructed topology as physical
# groups named ver#, edge#, face#, and face k is raster cell k, so:
#
#   grain boundary  : 1D element set "edge#", touching two "face#" sets
#   specimen surface: 1D element set touching one face
#   theta           : disorientation of the two grains' orientations
#                     (${STEM}-grainori.txt), computed in the Python driver
#   triple junction : mesh vertex where 3+ distinct edge ids meet
#
# Outputs, all named ${STEM}:
#   ${STEM}.msh4          Gmsh v4 mesh, linear triangles, all dimensions
#   ${STEM}.sttesr        raster geometry, used to derive the domain size
#   ${STEM}-grainori.txt  one orientation per grain, as read out of the tesr
#   check-ori.png         raster coloured by per-voxel orientation (IPF-z),
#                         border trimmed, scale bar added (micrograph.py)
#   check-grains.png      raster coloured by cell id, likewise
#   check-mesh.png        raster cells with every reconstructed boundary edge
#                         of the mesh drawn on top (mesh_overlay.py)
#   ${STEM}-areachange.csv  per-grain area, rastered vs meshed, and the percent
#                         change between them (grain_area_change.py)
#   check-area.png        the grains coloured by that change, and its
#                         distribution
#
# The area table is the quality number for this stage: -tesrsmooth and the
# meshing both move the boundary, and the change in a grain's area is what that
# motion does to the bulk term of the transport problem. Stage 1's equivalent
# is printed by ctf_to_tesr.py (the RMS disorientation between a pixel and the
# single orientation its grain is given).
#
# All parameters arrive as environment variables so the Python driver stays the
# single source of truth. Run standalone by exporting them yourself.
#
# Docs: -T https://neper.info/doc/neper_t.html
#       -M https://neper.info/doc/neper_m.html
#       keys https://neper.info/doc/exprskeys.html
# =============================================================================
set -euo pipefail

: "${TESR:?set TESR to the EBSD raster tessellation (.tesr)}"
: "${NEPER_BIN:=neper}"
: "${GMSH_BIN:=gmsh}"
: "${POVRAY_BIN:=povray}"            # neper -V renders through this
: "${NEPER_ENV:=}"                   # bin dir of the neper environment, if any
# Make the neper environment's helper programs visible without activating it:
# neper -V spawns `povray` (and -M spawns gmsh) by name unless given a path.
# The caller's python3 is remembered first, as a last resort for the overlay.
CALLER_PYTHON=$(command -v python3 || true)
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

# check images (neper -V needs POV-Ray, mesh_overlay.py needs matplotlib; a
# failure here is reported, not fatal). PYTHON_BIN is whichever interpreter
# has numpy and matplotlib; the driver passes its own, and if that one lacks
# matplotlib the neper environment's python is tried before giving up.
: "${CHECK_IMAGES:=1}"
: "${PYTHON_BIN:=python3}"
: "${UNIT_NAME:=um}"                 # length unit of the raster, for scale bars
for candidate in "$PYTHON_BIN" ${NEPER_ENV:+"$NEPER_ENV/python"} ${CALLER_PYTHON:+"$CALLER_PYTHON"}; do
    if "$candidate" -c "import matplotlib" 2>/dev/null; then
        PYTHON_BIN="$candidate"
        break
    fi
done

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
# 0. Stage the raster.
#
# By default nothing is done to it: ctf_to_tesr.py already crops, fills holes,
# and writes cell ids contiguously from 1 with the origin at (0,0), and its
# grains are connected components so `rmsat` has nothing to remove. 
#Set TESR_TRANSFORM to a Neper transformation chain if the input needs work
# that the converter did not do, e.g.
#   TESR_TRANSFORM="crop(square(...)),rmsat,autocrop,resetorigin,renumber,resetcellid"
# and expect to need --no-voxel-ori due to the readback bug in Neper 5.0.0.
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
#     with AZtec/MTEX, and a bad segmentation shows up as speckle or as grains
#     that are obviously back-filled. 
# -----------------------------------------------------------------------------
if [ "$CHECK_IMAGES" = "1" ]; then
    for img in check-ori check-grains; do
        need "$img.png" || continue
        if [ "$img" = check-ori ]; then
            opts="-datavoxcol ori -datavoxcolscheme ipf"
        else
            opts=""
        fi
        # shellcheck disable=SC2086
        if "$NEPER_BIN" -V "${STEM}-raw.tesr" -povray "$POVRAY_BIN" $opts -print "$img"; then
            # neper -V frames the flat map in the middle of a 3D canvas; cut the
            # border away and add a scale bar (the trimmed width is LX)
            "$PYTHON_BIN" "$(dirname "$0")/micrograph.py" "$img.png" \
                --trim --width "$LX" --unit "$UNIT_NAME" \
                || echo "  WARNING: $img.png left untrimmed (Pillow/matplotlib missing?)" >&2
        else
            echo "  WARNING: neper -V failed for $img (POV-Ray missing?)" >&2
        fi
    done
fi

# -----------------------------------------------------------------------------
# 1. Mesh the raster. Linear triangles, Gmsh v4, all dimensions, because the
#    FESTIM side reads it with dolfinx.io.gmshio and needs the 1D element sets
#    intact (they carry the edge ids of the reconstructed boundary topology,
#    which select the network and index theta).
#
#    Neper reconstructs the interfaces, smooths them, then meshes the edges
#    and faces with the -rcl-derived length. -tmp must exist beforehand.
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

# -----------------------------------------------------------------------------
# 2. Mesh overlay: the raster cells with every edge# element set of the mesh
#    drawn over them, black between two grains and grey on the specimen
#    surface. This is the set of segments the driver can select a network
#    from; the driver draws the theta-filtered subset as check-network.png.
# -----------------------------------------------------------------------------
if [ "$CHECK_IMAGES" = "1" ] && need "check-mesh.png"; then
    "$PYTHON_BIN" "$(dirname "$0")/mesh_overlay.py" \
        "${STEM}-raw.tesr" "${STEM}.msh4" -o check-mesh.png --unit "$UNIT_NAME" \
        || echo "  WARNING: check-mesh.png not written (matplotlib missing?)" >&2
fi

# -----------------------------------------------------------------------------
# 3. How much the grains changed size. Compares the voxel count of every raster
#    cell with the summed triangle area of the corresponding mesh face, so it
#    measures -tesrsmooth and the meshing together -- there is no intermediate
#    file to separate them from. Set TESR_SMOOTH=none and a different STEM to
#    get the discretisation on its own, then difference the two csv files.
#
#    It also re-derives the face -> cell correspondence from triangle
#    centroids, which is the only check anywhere in the pipeline that Neper's
#    face k really is raster cell k. Everything downstream indexes theta by
#    face id, so a failure here is fatal rather than cosmetic; the script says
#    so and exits non-zero, and that is deliberately not swallowed.
# -----------------------------------------------------------------------------
AREA_PNG=()
if [ "$CHECK_IMAGES" = "1" ]; then
    AREA_PNG=(-o check-area.png)
fi
if need "${STEM}-areachange.csv"; then
    "$PYTHON_BIN" "$(dirname "$0")/grain_area_change.py" \
        "${STEM}-raw.tesr" "${STEM}.msh4" \
        --csv "${STEM}-areachange.csv" --unit "$UNIT_NAME" \
        ${AREA_PNG[@]+"${AREA_PNG[@]}"}
fi

# Neper writes one .geo/.msh pair per tessellation entity into -tmp and deletes
# them as it goes; anything left after a successful run is debris from an
# earlier failure. Worth looking at if it was -M that failed -- running gmsh on
# the offending .geo by hand is the usual way to find out why.
rmdir tmp 2>/dev/null || echo "  note: $WORKDIR/tmp is not empty (stale gmsh scratch)"

echo "ok: ${STEM}.msh4  (${NCELL} grains, domain ${LX} x ${LY})"

# -----------------------------------------------------------------------------
# The misorientations are exact: they come from the two grain orientations, and
# a section measures those as well as a volume does. What a section cannot see
# is the boundary plane normal (only its trace) and the out-of-plane paths.
#
# The consequence lands on connectivity, which is what the transport script
# measures. Percolation thresholds differ sharply between the two: the fraction
# of special boundaries needed to break up the random-boundary network is around
# 0.35 in 2D and 0.775-0.85 in 3D (Schuh, Minich & Kumar, Phil. Mag. 83 (2003)
# 711; Frary & Schuh, Phil. Mag. 85 (2005) 1123). So a THETA_MIN filter that
# fragments this network may well leave the corresponding 3D one intact, and the
# error is one-sided.
#
# The exception is a genuinely columnar microstructure -- thin films,
# electrodeposits -- where the 2D treatment is not an approximation at all.
# -----------------------------------------------------------------------------