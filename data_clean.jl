### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 12bfaf38-81f0-11eb-22d6-576d173e096b
using CSV

# ╔═╡ 847db8cc-81f0-11eb-06a2-3f879cc1e60a
using DataFrames

# ╔═╡ 25110d4e-81f0-11eb-3199-d71594d39a19


# ╔═╡ 4cdebaec-81f0-11eb-1cb4-b57c2c97af3a
og = CSV.read("wgm_chess_games.csv", DataFrame)

# ╔═╡ 9c34fa24-81f1-11eb-3c74-3d076211e3db
names_and_results = og[!,[:white_username,:black_username, :white_result, :black_result]]

# ╔═╡ fed6345e-81f1-11eb-2d6d-75d8179d6f63
function find_result(row)
	if row[:white_result] == "win"
		return [row[:white_username], row[:black_username]]
	elseif row[:black_result] == "win"
		return [row[:black_username], row[:white_username]]
	else
		return nothing
	end
end

# ╔═╡ 521c8f5a-81f2-11eb-035b-6713df3c5122


# ╔═╡ b216fe40-81f2-11eb-3724-57a5cbef2c81
results = find_result.(eachrow(names_and_results))

# ╔═╡ 4b648134-8375-11eb-3608-1f697f35ac2d
games = hcat(filter!(x-> x!=nothing,results)...)

# ╔═╡ ba96213a-81f3-11eb-3302-ff5cbd8280eb
names = unique(vec(games))

# ╔═╡ 55082878-8378-11eb-2fa9-8ba4c37b6625
lower_names = lowercase.(names)

# ╔═╡ 9084d5cc-8378-11eb-00ff-6d2ddbaa588a
length(unique(lower_names)) == length(names)

# ╔═╡ eaaa5e22-81f3-11eb-1673-61c3a96bd15f
function replace_name(name)
	name_idx = findfirst(isequal(name),names)
	return name_idx
end

# ╔═╡ 73f57e3a-81f6-11eb-3384-1f4af03d702c
W = replace_name.(games)

# ╔═╡ 929e5aac-8377-11eb-2b81-4153e021ac50
findfirst(i -> i=="camillab", lower_names)

# ╔═╡ 6c60363c-8378-11eb-0c1c-2b9fc7a0f4de
g_df = DataFrame(collect(W'), [:winner,:loser])

# ╔═╡ 30bbc53a-8376-11eb-25a5-a59472220667
n_df = DataFrame([lower_names], [:name])

# ╔═╡ df5ff560-81f6-11eb-27b6-9bde56dbd5ce
CSV.write("games.csv", g_df)

# ╔═╡ d7118b60-8379-11eb-04e2-13025b8acea5
CSV.write("names.csv", n_df)

# ╔═╡ 558fa0dc-81f7-11eb-2e8c-8f5c8ed6aca2
# CSV.write("names.csv",DataFrame([nn]))

# ╔═╡ Cell order:
# ╠═12bfaf38-81f0-11eb-22d6-576d173e096b
# ╠═847db8cc-81f0-11eb-06a2-3f879cc1e60a
# ╟─25110d4e-81f0-11eb-3199-d71594d39a19
# ╠═4cdebaec-81f0-11eb-1cb4-b57c2c97af3a
# ╠═9c34fa24-81f1-11eb-3c74-3d076211e3db
# ╠═fed6345e-81f1-11eb-2d6d-75d8179d6f63
# ╟─521c8f5a-81f2-11eb-035b-6713df3c5122
# ╠═b216fe40-81f2-11eb-3724-57a5cbef2c81
# ╠═4b648134-8375-11eb-3608-1f697f35ac2d
# ╠═ba96213a-81f3-11eb-3302-ff5cbd8280eb
# ╠═55082878-8378-11eb-2fa9-8ba4c37b6625
# ╠═9084d5cc-8378-11eb-00ff-6d2ddbaa588a
# ╠═eaaa5e22-81f3-11eb-1673-61c3a96bd15f
# ╠═73f57e3a-81f6-11eb-3384-1f4af03d702c
# ╠═929e5aac-8377-11eb-2b81-4153e021ac50
# ╠═6c60363c-8378-11eb-0c1c-2b9fc7a0f4de
# ╠═30bbc53a-8376-11eb-25a5-a59472220667
# ╠═df5ff560-81f6-11eb-27b6-9bde56dbd5ce
# ╠═d7118b60-8379-11eb-04e2-13025b8acea5
# ╠═558fa0dc-81f7-11eb-2e8c-8f5c8ed6aca2
