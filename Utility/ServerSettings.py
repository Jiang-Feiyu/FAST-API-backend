class ServerSettings:

    MOUNT_FOLDER = "Data"

    def __init__(self):
        self.MOUNT_FOLDER = "Data"
        self.MESSAGE_FOLDER = "Data/messages"

    def get_mount_folder(self):
        return self.MOUNT_FOLDER

    def get_message_folder(self):
        return self.MESSAGE_FOLDER