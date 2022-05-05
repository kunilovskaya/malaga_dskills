#!/bin/sh

# remember the PID (e.g [7] 16403) to kill (kill -9 16403) the process when you want to stop watching the bib

# How-To:
# (1) edit /home/u2/Documents/bibs/manage/cleanbib.sh
# (2) run u2@MAK:~$ fswatch -o /home/u2/Documents/bibs/0_translated language.bib | xargs -n1 /home/u2/Documents/bibs/manage/cleanbib.sh &

# Call the python script
python3 /home/u2/Documents/bibs/manage/cleanbib.py -f /home/u2/Documents/bibs/0_translated_language.bib -o /home/u2/Documents/bibs/cleaned/0_translated_language.bib
echo "Cleaning finished. Copying ..."
# Copy the result to all projects:
cp /home/u2/Documents/bibs/cleaned/0_translated_language.bib /home/u2/texstudio/phd/refs/translated_lang.bib
cp /home/u2/Documents/bibs/cleaned/0_translated_language.bib /home/u2/texstudio/2020/transregisters/translated_lang.bib

echo "Done!"
