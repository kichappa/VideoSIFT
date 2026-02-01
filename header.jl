struct potential_blob
	thisX::UInt32 # 1
	y::UInt32 # 2
	thisImg::UInt32 # 3
	x::UInt32 # 4
	oct::UInt32 # 5
	lay::UInt32 # 6
end

potential_blob() = potential_blob(0, 0, 0, 0, 0, 0)
