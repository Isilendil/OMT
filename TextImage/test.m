
similarity = [1,2,3,4,5,6,7,8,9];
K = 3;

for i = 1 : K
	max_value = 0;
	max_index = 1;
	for j = 1 : length(similarity)
		if (similarity(j) > max_value)
			max_value = similarity(j);
			max_index = j;
		end
	end
	max_value_vector(i) = similarity(max_index);
	max_index_vector(i) = max_index;
  similarity(max_index) = 0;
end

similarity = [1,2,3,4,5,6,7,8,9];
for i = 1 : K
	min_value = 1000000;
	min_index = 1;
	for j = 1 : length(similarity)
		if (similarity(j) < min_value)
			min_value = similarity(j);
			min_index = j;
		end
	end
	min_value_vector(i) = similarity(min_index);
	min_index_vector(i) = min_index;
  similarity(min_index) = 1000000;
end

