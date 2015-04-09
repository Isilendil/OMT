function preprocess(data_file, similarity_file)

% load dataset
load(sprintf('old/%s', data_file));

% image data
image_num = size(image_fea, 1);

image_index = randperm(image_num);

image_X = image_fea(image_index, :);
image_Y = image_gnd(image_index, :);

for i = 1 : image_num
	  if (image_Y(i) == 0)
			  image_Y(i) = -1;
		end
end

image_fea = image_X;
image_gnd = image_Y;

% text data
text_num = size(text_fea, 1);

text_index = randperm(text_num);

text_X = text_fea(text_index, :);
text_Y = text_gnd(text_index, :);

for i = 1 : text_num
	  if (text_Y(i) == 0)
			  text_Y(i) = -1;
		end
end

text_fea = text_X;
text_gnd = text_Y;

save(data_file, 'image_fea', 'image_gnd', 'text_fea', 'text_gnd', 'co_image_fea', 'co_text_fea');

calculate_affinity(data_file, similarity_file);

end
