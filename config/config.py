import os

class config_container(object):

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            if value:
                yield value

    def __getattr__(self, item):
        return None

    def _to_dict(self):
        d = {}
        for attr, value in self.__dict__.iteritems():
            if isinstance(value, config_container):
                d[attr] = value._to_dict()
            else:
                d[attr] = value
        return d

    def __repr__(self):
        import json
        return json.dumps(self._to_dict(), indent=2)

def speaker_config():
    config = config_container()
    config.batch_size  = 10
    config.time_dim = 300
    config.freq_dim = 512
    config.num_speakers = 1251
    config.keep_prob = 0.80
    config.model = "speaker_recognition.SpeakerRecognition"
    config.save_path = "best_save/"
    config.load_mode = "continue"
    return config

def config():
    config = config_container()
    config.speaker_reco = speaker_config()
    return config


