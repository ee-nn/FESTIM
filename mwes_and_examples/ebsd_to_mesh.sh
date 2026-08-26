#!/usr/bin/env bash
# =============================================================================
# A single EBSD map -> fitted 2D tessellation -> triangular mesh, for the
# FESTIM codim-1 grain-boundary transport script.
#
# In 2D the tessellation cells *are* the faces, so the grain boundaries are
# edges and the triple junctions are vertices. Everything the transport script
# needs therefore comes out of `-statedge` and `-statver` rather than
# `-statface` and `-statedge`:
#
#   grain boundary  : interior edge      (theta, length live here)
#   triple junction : interior vertex with 3+ edges
#   quadruple node  : does not exist in 2D
#
# This is still "route B": a convex-cell (Laguerre) tessellation is *fitted* to
# the raster and the mesh is built from that. Route A -- `neper -M map.tesr`,
# which is available in 2D and only in 2D -- is not used, because it produces
# no .tess and therefore no `-statedge`, hence no theta and no way to tell a
# grain boundary from a piece of specimen surface.
#
# Outputs, all named ${STEM}:
#   ${STEM}.tess          regularized tessellation (the one that was meshed)
#   ${STEM}.msh4          Gmsh v4 mesh, linear triangles
#   ${STEM}.stedge        domtype domedge theta length ymin ymax  (per edge)
#   ${STEM}.stver         domtype edgenb                          (per vertex)
#   ${STEM}.sttesr        raster geometry, used to derive the domain size
#   ${STEM}-grainori.txt  one orientation per grain, as read out of the tesr
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
: "${STEM:=poly}"
: "${WORKDIR:=.}"
: "${FORCE:=0}"

: "${CRYSYM:=cubic}"
: "${ORIDES:=rodrigues:passive}"      # must match how the tesr stores them

# Optional Neper transformation chain applied to the input raster. Empty by
# default: the converter is expected to have done the cropping and cleanup, and
# skipping this avoids Neper's tesr write path (see stage 0).
: "${TESR_TRANSFORM:=}"

# tessellation fitting. The objective is `avdiameq * rms(distance)` in the
# tesr's absolute length unit, so `val` and `eps` are dimensional: keep the
# raster in microns (ctf_to_tesr.py's default) and never use `val<1e-6` with a
# metre-scale raster, where the initial objective (~1e-10) already satisfies it
# and Neper returns the initial Laguerre guess after one iteration.
: "${OBJ_RES:=8}"                     # control points per grain per direction
: "${MORPHO_STOP:=eps<1e-6||val<1e-12||iter>=20000||time>=3600}"
# Algorithms tried in order after a plateau (Neper retries the current one
# once, then moves on, then gives up, keeping the best solution). Neper's
# default; see MORPHO_ALGO in the driver before changing it, and avoid a
# single-entry list.
: "${MORPHO_ALGO:=subplex,praxis}"
: "${RSEL:=0.8}"                      # small-edge length for regularization

# meshing. In 2D the cells are faces, so -rcl sets the element size inside the
# grains; the grain boundaries are edges (-rcledge) and the triple junctions
# are vertices (-rclver).
: "${RCL:=0.8}"
: "${RCL_EDGE:=0.2}"
: "${RCL_VER:=}"
: "${PL:=2.5}"
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
# grains are connected components so `rmsat` has nothing to remove. Copying
# rather than passing the file through `neper -T -transform` is deliberate --
# Neper 5.0.0 writes a raster tessellation it cannot read back when the file
# carries a `**oridata` section, so any transform here yields a poly-raw.tesr
# that stalls forever on the next parse.
#
# Set TESR_TRANSFORM to a Neper transformation chain if the input needs work
# that the converter did not do, e.g.
#   TESR_TRANSFORM="crop(square(...)),rmsat,autocrop,resetorigin,renumber,resetcellid"
# and expect to need --no-voxel-ori on the converter for the reason above.
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
# `rastersize*` is voxnb* x voxsize*, i.e. the physical extent. There is no
# cell-count key for a tesr, so the grain count is the line count of the
# orientation file below.
"$NEPER_BIN" -T -loadtesr "${STEM}-raw.tesr" \
    -stattesr dim,rastersizex,rastersizey,voxsizex,voxsizey \
    -o "${STEM}"

read -r DIM LX LY VSX VSY < "${STEM}.sttesr"
echo "  raster: dim=$DIM  extent=${LX} x ${LY}  pixel=${VSX} x ${VSY}"
# Everything below runs in the raster's unit; the Python driver converts the
# mesh and the .st* lengths to metres (TESR_UNIT) after reading them.

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
# 1. Fit the tessellation.
#
# -n from_morpho takes the cell count from the raster. The objective function
# samples control points on the grain boundaries and minimises the distance
# between raster and tessellation cell boundaries; res is the number of control
# points per grain per direction (Neper's default is 5).
#
# 2D is the cheap case: with -morphooptidof x,y,w there are three degrees of
# freedom per grain instead of four, and the cells are polygons rather than
# polyhedra, so a few hundred grains fit in minutes rather than hours. Crop
# anyway if the map has thousands -- the cost is superlinear.
#
# The orientations are attached here rather than afterwards: `-ori from_morpho`
# reads them from `-morphooptiini ori:file(...)`, so cell k of the tessellation
# carries line k of the file, which is cell k of the raster. Getting this wrong
# is silent -- the geometry is unaffected and only `theta` goes wrong -- so the
# Python driver cross-checks the count and the disorientation distribution.
# -----------------------------------------------------------------------------
# Check `Initial solution: f = ...` and the iteration count in the log: at
# micron scale the initial value is O(10-100) and the fit should run for
# minutes. A one-iteration exit on `val' means the raster unit is wrong.
if need "${STEM}-fit.tess"; then
    "$NEPER_BIN" -T -n from_morpho \
        -dim 2 \
        -domain "square($LX,$LY)" \
        -morpho "tesr:file(${STEM}-raw.tesr)" \
        -morphooptiobjective "tesr:pts(region=surf,res=${OBJ_RES})+val(bounddist)" \
        -morphooptidof x,y,w \
        -morphooptialgo "$MORPHO_ALGO" \
        -morphooptistop "$MORPHO_STOP" \
        -morphooptilogval iter,val \
        -crysym "$CRYSYM" \
        -oridescriptor "$ORIDES" \
        -ori from_morpho \
        -morphooptiini "ori:file(${STEM}-grainori.txt,des=${ORIDES})" \
        -o "${STEM}-fit"
fi

# -----------------------------------------------------------------------------
# 2. Regularize, and write the statistics *from the regularized tessellation*.
#
# This ordering is not cosmetic. Regularization deletes small edges, so edge ids
# are renumbered. The mesh is built from the regularized tess and the 1D element
# sets in the .msh4 carry *its* edge ids, so the .stedge used to interpret those
# tags has to come from the same file, not from the fitted one.
#
# -rsel is relative to the average cell size; a value of 1 corresponds to a
# length of 0.125 for a unit-area cell in 2D. It should track whatever -rcl is
# actually used, which is why the Python driver sets both from one number.
#
# `domtype` is 0/1 for an entity on a domain vertex/edge and negative when the
# entity is interior; `domedge` is the id of the domain edge an edge lies on, or
# -1. In 2D the domain boundary is made of edges, so either column identifies
# specimen surface vs real grain boundary -- both are written so the Python side
# can cross-check them against each other rather than trusting one convention.
# -----------------------------------------------------------------------------
if need "${STEM}.tess"; then
    "$NEPER_BIN" -T -loadtess "${STEM}-fit.tess" \
        -reg 1 -rsel "$RSEL" \
        -statedge domtype,domedge,theta,length,ymin,ymax \
        -statver domtype,edgenb \
        -o "$STEM"
fi

# -----------------------------------------------------------------------------
# 3. Mesh. Linear triangles, Gmsh v4, because the FESTIM side reads it with
#    dolfinx.io.gmshio and needs the 1D element sets to survive intact -- they
#    are what carry the tessellation edge ids that select the network.
#
#    -dim all is redundant (the .msh always holds every dimension unless :msh
#    is appended) but makes the intent explicit: the 1D mesh is not a by-product
#    here, it is the object of interest.
# -----------------------------------------------------------------------------
if need "${STEM}.msh4"; then
    "$NEPER_BIN" -M "${STEM}.tess" \
        -gmsh "$GMSH_BIN" \
        -dim all \
        -order 1 \
        -elttype tri \
        -rcl "$RCL" \
        -rcledge "$RCL_EDGE" \
        ${RCL_VER:+-rclver "$RCL_VER"} \
        -pl "$PL" \
        ${MESH_QUAL_MIN:+-meshqualmin "$MESH_QUAL_MIN"} \
        ${MESH_MAX_TIME:+-mesh2dmaxtime "$MESH_MAX_TIME"} \
        -tmp tmp \
        -format msh4 \
        -statmesh nodenb,eltnb \
        -o "$STEM"
fi

# Neper writes one .geo/.msh pair per tessellation entity into -tmp and deletes
# them as it goes; anything left after a successful run is debris from an
# earlier failure. Worth looking at if it was -M that failed -- running gmsh on
# the offending .geo by hand is the usual way to find out why.
rmdir tmp 2>/dev/null || echo "  note: $WORKDIR/tmp is not empty (stale gmsh scratch)"

echo "ok: ${STEM}.msh4  (${NCELL} grains, domain ${LX} x ${LY})"

# -----------------------------------------------------------------------------
# What 2D costs, so it is a choice rather than an accident
#
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