--Split a string in an array of string
function string:split(sep)
        local sep, fields = sep or ":", {}
        local pattern = string.format("([^%s]+)", sep)
        self:gsub(pattern, function(c) fields[#fields+1] = c end)
        return fields
end

--Load a text file in a Tensor object
function readData(file)
	io.input(file)
	local nb_lines = 0
	local n = 0

	for line in io.lines() do
		nb_lines = nb_lines+1
		if nb_lines == 1 then
			n = table.getn(line:split(" "))
		end
	end

	io.input(file)

	local data = torch.Tensor(nb_lines, n)

	local i = 1
	for line in io.lines() do
		local v = line:split(" ")
		for j = 1,n do
			data[i][j] = v[j]
		end
		i = i+1
	end
	return data
end