import os
import time
import pygame
import threading
import webbrowser
import tkinter.messagebox

import numpy as np

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from mutagen.mp3 import MP3
from PIL import Image, ImageTk
from ttkthemes import themed_tk as tk
from model_utils.utils import istft_reconstruction, ZPSPBM, to_byte
from model_utils.model_inference import run_inference



class PlayThread(threading.Thread):
    def __init__(self, callback=None, callback_args=None, *args, **kwargs):
        target = kwargs.pop('target')
        super(PlayThread, self).__init__(target=self.target_with_callback, *args, **kwargs)
        self.callback = callback
        self.method = target
        self.callback_args = callback_args

    def target_with_callback(self):
        self.method()
        if self.callback is not None:
            self.callback(*self.callback_args)


def audio_process_thread():
    print("thread start successfully!")
    '''
    Initializing the mixer
    '''
    # pre_init and set buffer size
    stft_hop = 512
    stft_frame = 1022
    audRate = 44100
    spec_buff_size = int(buff_size / stft_hop)
    pygame.mixer.pre_init(audRate // 2, -16, 2, buff_size)
    pygame.mixer.init()
    mag_remix = mag_mix + (volume - 0.5) * pred_mag
    frame_wav_remix = istft_reconstruction(mag_remix[:, 0:spec_buff_size + 2],
                                           phase_mix[:, 0:spec_buff_size + 2],
                                           hop_length=stft_hop, win_length=stft_frame)
    frame_wav_remix = frame_wav_remix[:1 * buff_size]
    byte_wav_remix = to_byte(frame_wav_remix)
    data = b''.join(byte_wav_remix)

    sound = pygame.mixer.Sound(data)
    chan = pygame.mixer.find_channel()
    chan.queue(sound)
    i = 1

    t_end = time.time() + playback_time
    while time.time() < t_end:
        if stopped:
            playing = False

            Elapsed_time['text'] = "Timer : 00:00"

            pygame.mixer.quit()
            break
        time.sleep(0.0001)
        if chan.get_queue() is None:
            mag_remix = mag_mix + (volume - 0.5) * pred_mag
            frame_wav_remix = istft_reconstruction(mag_remix[:, i * spec_buff_size:(i + 1) * spec_buff_size + 2],
                                                   phase_mix[:, i * spec_buff_size:(i + 1) * spec_buff_size + 2],
                                                   hop_length=stft_hop, win_length=stft_frame)
            frame_wav_remix = frame_wav_remix[:1 * buff_size]
            byte_wav_remix = to_byte(frame_wav_remix)
            data = b''.join(byte_wav_remix)
            sound = pygame.mixer.Sound(data)
            chan.queue(sound)
            i = i + 1

    print("thread ended successfully!")


def cb(param1, param2):
    global playing
    # this is run after your thread end
    playing = False
    print ("{} {}".format(param1, param2))


'''
global params
'''
# buffer size (samples)
buff_size = 4096
# fix playback_time (secs)
playback_time = 5
# create empty play list
playlist = []
# init playing status
playing = False
stopped = True
# init volumes
volume = 0.75
volume_2 = 0.75

# Setting up the ttk themes
root = tk.ThemedTk()
root.get_themes()
root.set_theme("equilux")
# making the window Un-Resizable
root.resizable(0, 0)
# Creating and packing the status bar
Status_bar = ttk.Label(root, text="Remixing Music with Visual Conditioning", relief=SUNKEN, anchor=W)
Status_bar.pack(fill=X, side=BOTTOM)
# Creating the left and Rigth frames
Left_frame = ttk.Frame(root)
Right_frame = ttk.Frame(root)
Left_frame.pack(side=LEFT, fill=X and Y)
Right_frame.pack(side=RIGHT, fill=X and Y)
# Creating the List box
listbox = Listbox(Left_frame)
listbox.pack(padx=30, pady=30)
# Creating some Frames which are located in the Right frame
Top_frame = ttk.Frame(Right_frame)
Middle_frame = ttk.Frame(Right_frame)
Low_frame = ttk.Frame(Right_frame)
Bottom_frame = ttk.Frame(Right_frame)
Top_frame.pack(side=TOP, fill=X)
Middle_frame.pack(side=TOP, fill=X, pady=5)
Low_frame.pack(side=TOP, fill=X, pady=5)
Bottom_frame.pack(side=TOP, fill=X, pady=5)
# Customizing the Root window
root.title("VAreMixer: Remixing Music with Visual Conditioning")
root.configure(bg="#373737")
# root window size
root.geometry('900x360')
# Creating the Menu bar
main_menu = Menu(root)
root.config(menu=main_menu)
file_menu = Menu(main_menu, tearoff=0)
main_menu.add_cascade(label="File", menu=file_menu)

'''
Creating a function to add files to the playlist
'''


def ask_open():
    global file_path
    x = filedialog.askopenfilename(initialdir=os.getcwd()+'/test_case')
    add_to_playlist(x)


def add_to_playlist(file_path):
    dir_name = os.path.basename(file_path)
    index = 0
    listbox.insert(index, dir_name)
    playlist.insert(index, file_path)
    index += 1


def ask_del():
    selected_song = listbox.curselection()
    selected_song = int(selected_song[0])
    listbox.delete(selected_song)
    playlist.pop(selected_song)


# Adding buttons to the Lists
add_butt = ttk.Button(Left_frame, text="+ ADD", command=ask_open)
del_butt = ttk.Button(Left_frame, text="- DEL", command=ask_del)
add_butt.pack(side=LEFT, padx=10)
del_butt.pack(side=LEFT)

file_menu.add_command(label="Open", command=ask_open)
file_menu.add_command(label="Exit", command=root.destroy)

help_menu = Menu(main_menu, tearoff=0)
main_menu.add_cascade(label="Help", menu=help_menu)


def about_us():
    a = tkinter.messagebox.askquestion("Inference Demo for the paper: Remixing Music with Visual Conditioning",
                                       "Author: Li-Chia (Richard) Yang"
                                       "\nVisit project repo?")
    if a == "yes":
        x = ""
        url = "https://github.com/RichardYang40148/VAreMixer"

        webbrowser.open(url, new=x)

help_menu.add_command(label="More about the project ...", command=about_us)

'''
Creating a function to calculate the sound Length
'''


def show_length():
    mins, secs = divmod(playback_time, 60)
    # Rounding the Minutes(mins) and Seconds(secs)
    mins = round(mins)
    secs = round(secs)

    format_the_time = '{:02d}:{:02d}'.format(mins, secs)
    Tot_len['text'] = "Audio Length : " + format_the_time

    Helper = threading.Thread(target=count_tracker, args=(playback_time,))
    Helper.start()


def count_tracker(total_length):
    global stopped
    current_time = 0

    while current_time <= total_length and stopped is False:
        mins, secs = divmod(current_time, 60)
        mins = round(mins)
        secs = round(secs)

        format_the_time = '{:02d}:{:02d}'.format(mins, secs)
        Elapsed_time['text'] = "Timer : " + format_the_time
        time.sleep(1)
        current_time += 1


def instrument_file_assignment(inst_num):
    inst_dict = {'1': 'demo_utils/instrument_pics/accordion_0.jpg','2': 'demo_utils/instrument_pics/acoustic_guitar_0.jpg',
                 '3': 'demo_utils/instrument_pics/cello_0.jpg',  '4':'demo_utils/instrument_pics/clarinet_0.jpg',
                 '5': 'demo_utils/instrument_pics/flute_0.jpg', '6': 'demo_utils/instrument_pics/violin_0.jpg',
                 '7': 'demo_utils/instrument_pics/trumpet_0.jpg', '8': 'demo_utils/instrument_pics/xylophone_0.jpg'}
    return inst_dict[inst_num]

'''
create Play, load, and stopbutton
'''

def load_button():
    global mix_wav
    global pred_mag
    global mag_mix
    global phase_mix
    global inst_num
    
    try:
        selecting_the_selected_file = listbox.curselection()
        selecting_the_selected_file = int(selecting_the_selected_file[0])
        input_file = playlist[selecting_the_selected_file]
        inst_file = instrument_file_assignment(str((inst_num.get())))
        run_inference(input_file, inst_file)
        model_output_dir = 'model_utils/ckpt/model/visualization/output'
        pred_mask = np.load(model_output_dir + "/pred_mask1.npy")
        phase_mix = np.load(model_output_dir + "/mix_phase.npy")
        mag_mix = np.load(model_output_dir + "/mix_mag.npy")
        pred_zpspbm = ZPSPBM(pred_mask)
        
        pred_mag = mag_mix * pred_zpspbm
        
        Status_bar['text'] = "Done loading!"
        Playing_demo['text'] = os.path.basename(input_file)
    except NameError:
        tkinter.messagebox.showerror("File not found",
                                     "The file you selected is not found or you have not selected a file")
    except IndexError:
        tkinter.messagebox.showerror("File not found",
                                     "The file you selected is not found or you have not selected a file")
    except KeyError:
            tkinter.messagebox.showerror("Instrument image not selected",
                                         "Please setect an instrument")

def play_button():
    global stopped
    global playing
    try:
        stopped = False
        selecting_the_selected_file = listbox.curselection()
        selecting_the_selected_file = int(selecting_the_selected_file[0])
        play_file = playlist[selecting_the_selected_file]
        Playing_demo['text'] = os.path.basename(play_file)
        Status_bar['text'] = "Playing" + " : " + os.path.basename(play_file)
        if not playing:
            show_length()
            thread = PlayThread(name='audio_process_thread',
                                target=audio_process_thread,
                                callback=cb,
                                callback_args=("thread", "callback")
                                )
            thread.start()
            playing = True
    except NameError:
        tkinter.messagebox.showerror("File not found",
                                     "The file you selected is not found or you have not selected a file")
    except IndexError:
        tkinter.messagebox.showerror("File not found",
                                     "The file you selected is not found or you have not selected a file")


def stop_button():
    global stopped
    Status_bar['text'] = "Stopped"
    stopped = TRUE


# Creating labels
Playing_demo = ttk.Label(Top_frame, text="Select an audio file and target instrument...")
Tot_len = ttk.Label(Top_frame, text="")
Elapsed_time = ttk.Label(Top_frame, text="")
Playing_demo.pack(side=TOP, fill=X, pady=10)
Tot_len.pack(side=TOP, fill=X, pady=10)
Elapsed_time.pack(side=TOP, fill=X, pady=10)

s = ttk.Style()
s.layout('TRadiobutton',
         [('Radiobutton.padding',
           {'children':
            [('Radiobutton.indicator', {'side': 'bottom', 'sticky': ''}), # Just need to change indicator's 'side' value
             ('Radiobutton.focus', {'side': 'left',
                                    'children':
                                    [('Radiobutton.label', {'sticky': 'nswe'})],
                                    'sticky': ''})],
            'sticky': 'nswe'})])

# Creating a photoimage object to use image 
image_size = (64, 64)
photo1 = ImageTk.PhotoImage(Image.open('demo_utils/instrument_pics/accordion_0.jpg').resize(image_size))
photo2 = ImageTk.PhotoImage(Image.open('demo_utils/instrument_pics/acoustic_guitar_0.jpg').resize(image_size))
photo3 = ImageTk.PhotoImage(Image.open('demo_utils/instrument_pics/cello_0.jpg').resize(image_size))
photo4 = ImageTk.PhotoImage(Image.open('demo_utils/instrument_pics/clarinet_0.jpg').resize(image_size))
photo5 = ImageTk.PhotoImage(Image.open('demo_utils/instrument_pics/flute_0.jpg').resize(image_size))
photo6 = ImageTk.PhotoImage(Image.open('demo_utils/instrument_pics/violin_0.jpg').resize(image_size))
photo7 = ImageTk.PhotoImage(Image.open('demo_utils/instrument_pics/trumpet_0.jpg').resize(image_size))
photo8 = ImageTk.PhotoImage(Image.open('demo_utils/instrument_pics/xylophone_0.jpg').resize(image_size))
  # here, image option is used to 
# set image on button 
inst_num =  IntVar()
inst1_butt = ttk.Radiobutton(Top_frame, image = photo1, variable=inst_num, value=1)
inst2_butt = ttk.Radiobutton(Top_frame, image = photo2, variable=inst_num, value=2)
inst3_butt = ttk.Radiobutton(Top_frame, image = photo3, variable=inst_num, value=3)
inst4_butt = ttk.Radiobutton(Top_frame, image = photo4, variable=inst_num, value=4)
inst5_butt = ttk.Radiobutton(Top_frame, image = photo5, variable=inst_num, value=5)
inst6_butt = ttk.Radiobutton(Top_frame, image = photo6, variable=inst_num, value=6)
inst7_butt = ttk.Radiobutton(Top_frame, image = photo7, variable=inst_num, value=7)
inst8_butt = ttk.Radiobutton(Top_frame, image = photo8, variable=inst_num, value=8)

inst1_butt.pack(side=LEFT, padx=5)
inst2_butt.pack(side=LEFT, padx=5)
inst3_butt.pack(side=LEFT, padx=5)
inst4_butt.pack(side=LEFT, padx=5)
inst5_butt.pack(side=LEFT, padx=5)
inst6_butt.pack(side=LEFT, padx=5)
inst7_butt.pack(side=LEFT, padx=5)
inst8_butt.pack(side=LEFT, padx=5)

# def out():
#     global var
#     # count = count + 1

#     answer = (var.get())
    
#     print (answer)
    
# test_button = ttk.Button(Top_frame,text = "Submit",command = out)
# test_button.pack(side=LEFT, padx=5)


# create buttons
load_butt = ttk.Button(Middle_frame, text='process', command=load_button)
load_butt.pack(side=LEFT, padx=5, pady=15)

stop_butt = ttk.Button(Middle_frame, text='stop', command=stop_button)
stop_butt.pack(side=RIGHT, padx=5, pady=15)

play_butt = ttk.Button(Middle_frame, text='play', command=play_button)
play_butt.pack(side=RIGHT, padx=5, pady=15)


'''
Creating the Volume scale
'''


def volume_scale(val):
    global volume
    volume = (float(val) - 50) / 100


def volume_scale_2(val):
    global volume_2
    volume_2 = (float(val) - 50) / 100


scale = ttk.Scale(Low_frame, from_=0, to=100, length=450, orient=HORIZONTAL, command=volume_scale)


Bottom_len = ttk.Label(Low_frame, text="Target source")


Bottom_len.pack(side=LEFT, padx=5, pady=15)
scale.set(75)
scale.pack(side=LEFT, padx=5, pady=15)

## For second mixer slider
# scale_2 = ttk.Scale(Bottom_frame, from_=0, to=100, length=300, orient=HORIZONTAL, command=volume_scale_2)
# Bottom_len_2 = ttk.Label(Bottom_frame, text="source 2")
# Bottom_len_2.pack(side=LEFT, padx=5, pady=15)
# scale_2.set(75)
# scale_2.pack(side=LEFT, padx=5, pady=15)

'''
Deleting the window while playing a song without any error
'''


def close_window():
    try:
        pygame.mixer.stop()
    except:
        pass
    root.destroy()


root.protocol("WM_DELETE_WINDOW", close_window)
root.mainloop()
