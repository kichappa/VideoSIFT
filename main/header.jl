struct potential_blob
	thisX::Int64 # 1
	y::Int32 # 2
	thisImg::Int32 # 3
	x::Int64 # 4
	oct::Int32 # 5
	lay::Int32 # 6
end

potential_blob() = potential_blob(0, 0, 0, 0, 0, 0)
