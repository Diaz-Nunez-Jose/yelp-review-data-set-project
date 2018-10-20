import os
path = 'category-txt-files'
write_path = 'cleaned-up-category-txt-files/'

for subdir, dirs, files in os.walk(path):
	for file in files:
		file_path = subdir + os.path.sep + file
		with open(file_path, "r", encoding = "utf-8") as text:
			cleaned_up_text = text.read()
			if cleaned_up_text[:2] == 'b\'' or cleaned_up_text[:2] == 'b"':
				cleaned_up_text = cleaned_up_text[2:]
			cleaned_up_text = cleaned_up_text.replace('"b"', ' ')
			cleaned_up_text = cleaned_up_text.replace('\'b\'', ' ')
			cleaned_up_text = cleaned_up_text.replace('"b\'', ' ')
			cleaned_up_text = cleaned_up_text.replace('\'b"', ' ')
			cleaned_up_text = cleaned_up_text[:len(cleaned_up_text) - 1]
			with open(write_path + file, "w", encoding = "utf-8") as write_to_file:
				write_to_file.write(cleaned_up_text)