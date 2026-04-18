"""
    load_material(filename; material_name=nothing) → UniversalMaterial

Parse an Abaqus `.inp` file and construct a fully type-specialised `UniversalMaterial`.

The function reads `*PARAMETER TABLE, type="UNIVERSAL_TAB"` blocks, whose rows have
the format:
    kInv, kf0, kf1, kf2, w0, w1, w2

where:
  - `kInv`      – invariant index (1–15)
  - `kf0/1/2`   – layer function type for h₀, h₁, h₂
  - `w0/w1/w2`  – layer weights

# Arguments
- `filename`      – path to the `.inp` file
- `material_name` – (optional) name matching a `*MATERIAL, name=...` block;
                    when omitted, the first `UNIVERSAL_TAB` block in the file is used

# Notes
- `*INCLUDE` directives are silently skipped (the `UNIVERSAL_PARAM_TYPES.inc` schema
  file is not required)
- `Nfibers` is inferred from the maximum `kInv` used:
    - `kInv ≤ 3`  → 0 fibers (isotropic)
    - `kInv ≤ 5`  → 1 fiber family
    - `kInv ≤ 9`  → 2 fiber families
    - `kInv ≤ 15` → 3 fiber families
- The returned type has the full network topology encoded in type parameters.
  The first call to `evaluate` triggers compilation of a specialised, unrolled
  `strain_energy` function for that topology.
"""
function load_material(filename::String; material_name::Union{String, Nothing}=nothing)
    rows = parse_universal_tab(filename, material_name)
    isempty(rows) && error(
        "No UNIVERSAL_TAB data found in \"$filename\"" *
        (material_name === nothing ? "" : " for material \"$material_name\""),
    )
    return build_material(rows)
end

"""
    parse_universal_tab(filename, material_name) -> Vector{NTuple{7, Float64}}

Parses the given `.inp` file and extracts the rows of the `UNIVERSAL_TAB` parameter table
corresponding to the specified material name (or the first one if `material_name` is `nothing`).
"""
function parse_universal_tab(filename::String,material_name::Union{String, Nothing}) :: Vector{NTuple{7, Float64}}
    rows   = NTuple{7, Float64}[]
    target = material_name !== nothing ? uppercase(strip(material_name)) : nothing

    # When no name filter is given we accept the very first UNIVERSAL_TAB found.
    in_target = (target === nothing)
    in_table  = false

    for raw in eachline(filename)
        line = strip(raw)
        isempty(line)           && continue
        startswith(line, "**") && continue      # Abaqus full-line comment

        if startswith(line, "*")
            # Any Abaqus keyword ends the current data block.
            in_table = false
            upper    = uppercase(line)

            if startswith(upper, "*MATERIAL")
                # Track whether we are inside the requested material section.
                if target !== nothing
                    m = match(r"name\s*=\s*([^\s,]+)"i, line)
                    in_target = (m !== nothing && uppercase(m[1]) == target)
                end

            elseif in_target && startswith(upper, "*PARAMETER TABLE")
                in_table = occursin("UNIVERSAL_TAB", upper)
            end

        elseif in_table
            # Data line: kInv,kf0,kf1,kf2,w0,w1,w2
            parts = split(line, ",")
            length(parts) < 7 && continue
            try
                row = ntuple(i -> parse(Float64, strip(parts[i])), 7)
                push!(rows, row)
            catch
                # silently skip malformed lines
            end
        end
    end

    return rows
end