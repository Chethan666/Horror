# Hollow Lines (working title)

Hollow Lines is a faceless YouTube horror storytelling channel for a global audience, told in still pictures that look like frames of a dark horror film (real-looking people and places, low light, film grain and a single blood-red accent), each under a slow camera move, over voices and music, introduced each episode by a recurring lantern-carrying host called The Storyteller. About three-quarters of the stories are original, set in one shared fictional world, so recurring places, characters and hidden clues reward returning viewers who theorize across episodes. The rest retell folklore from around the world, which brings in search traffic. Each story is built for suspense rather than gore: a hook in the first ten seconds, everyday settings, rising dread, and endings that land as terror, a twist, justice or heartbreak. Every video is clearly presented as fiction or folklore and marked "not made for kids". The channel posts long episodes (8+ minutes) in English, with titles, descriptions and captions translated into six more languages, and joins finished episodes into long compilations; Shorts are switched off for now. Production is fully automated in two pipelines. The [story factory](STORY_FACTORY.md) is a writers' room of AI models from several companies (today open-weight models on Amazon Bedrock: Kimi, GLM, DeepSeek, Qwen and MiniMax) that pitch, plan, write, edit and audience-test every story, running in GitHub Actions. The [production pipeline](PRODUCTION.md) is a team of agents (preflight, casting, voice, storyboard, continuity, art director, cinematographer, illustrator and critic, art review, colorist, sound, edit, screening, packaging, QA) that voices the host, the teller and each character, paints every beat, scores it with an original score in stereo, assembles the video with FFmpeg and watches the cut before a person does, also in GitHub Actions.

## The two pipelines

| # | Pipeline | What it makes | Code | Docs |
|---|---|---|---|---|
| 1 | Story factory | horror stories for Hollow Lines | `story_factory/` | [STORY_FACTORY.md](STORY_FACTORY.md) |
| 2 | Video production | film-still videos (1440p, camera moves, atmosphere) from those stories | `production/` | [PRODUCTION.md](PRODUCTION.md) |

Both run in the same GitHub Actions workflow (one mode each) and share the model access,
the database and the cost log; each has its own stages and settings.

