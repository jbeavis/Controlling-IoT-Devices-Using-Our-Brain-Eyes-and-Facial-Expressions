import pygame
import os

# Initialize pygame
pygame.init()
pygame.mixer.init()

class MusicPlayer:
    def __init__(self, happy_music_dir = "Main/Music/Happy", sad_music_dir = "Main/Music/Sad"):
        print(os.getcwd())
        self.happy_music_dir = happy_music_dir
        self.sad_music_dir = sad_music_dir
        self.happyPlaylist = [f for f in os.listdir(happy_music_dir) if f.endswith(".mp3")]
        self.sadPlaylist = [f for f in os.listdir(sad_music_dir) if f.endswith(".mp3")]
        self.current_track_index = 0
        self.paused = False
        if not self.happyPlaylist: 
            print("No MP3 files found in the happy music directory.")
            exit()
        if not self.sadPlaylist:
            print("No MP3 files found in the sad music directory.")
            exit()
        self.emotion = "Happy"
        pygame.mixer.music.set_endevent(pygame.USEREVENT)  # Set event for track end, otherwise it won't continue to the next once one has finished
        self.load_track()
    
    def load_track(self):
        if self.emotion == "Happy":
            track_path = os.path.join(self.happy_music_dir, self.happyPlaylist[self.current_track_index])
            print(f"Loaded: {self.happyPlaylist[self.current_track_index]}")
        elif self.emotion == "Sad":
            track_path = os.path.join(self.sad_music_dir, self.sadPlaylist[self.current_track_index])
            print(f"Loaded: {self.sadPlaylist[self.current_track_index]}")
        pygame.mixer.music.load(track_path)
    
    def play(self):
        pygame.mixer.music.play()
        self.paused = False
        # print("Playing...")

    def pause(self):
        if not self.paused:
            pygame.mixer.music.pause()
            self.paused = True
            print("Paused")
        else:
            pygame.mixer.music.unpause()
            self.paused = False
            print("Resumed")

    def stop(self):  
        pygame.mixer.music.stop()
        print("Stopped playback")

    def next_track(self):
        self.current_track_index = (self.current_track_index + 1) % len(self.happyPlaylist)
        self.load_track()
        self.play()

    def prev_track(self):
        self.current_track_index = (self.current_track_index - 1) % len(self.happyPlaylist)
        self.load_track()
        self.play()

    def increase_volume(self):  
        volume = pygame.mixer.music.get_volume()
        pygame.mixer.music.set_volume(min(1.0, volume + 0.1))
        print(f"Volume increased to {pygame.mixer.music.get_volume():.1f}")

    def decrease_volume(self):  
        volume = pygame.mixer.music.get_volume()
        pygame.mixer.music.set_volume(max(0.0, volume - 0.1))
        print(f"Volume decreased to {pygame.mixer.music.get_volume():.1f}")
