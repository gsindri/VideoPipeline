# Privacy Policy — VideoPipelineStudio_actions

Last updated: 2026-02-09

This GPT ("VideoPipelineStudio_actions") uses a custom Action to call a self-hosted API ("VideoPipeline Studio Actions API") that runs on the builder's own machine. This policy describes how data is handled by that external API.

## What data is sent to the external API
When you use this GPT and it calls an Action, relevant parts of your input (for example: a video URL you provide, configuration options, and identifiers like project_id/job_id) are sent to the VideoPipeline Studio Actions API so it can perform the requested operation.

OpenAI may send only the information needed to perform the Action call. OpenAI does not audit or control how external APIs store or use your data.

## What the API does with that data
The API may:
- download the video/audio from the URL you provide (e.g., Twitch/YouTube),
- analyze it (transcription, chat analysis, highlight detection, export),
- return job status and results back to the GPT.

## Where data is stored
By default, results and intermediate files are stored locally on the machine running VideoPipeline Studio (the "operator machine"). This GPT's builder does not provide a separate cloud storage service for your data.

## Logging
The operator machine may keep local logs for debugging (for example, request timestamps, job IDs, and error messages).
If the API is exposed through Cloudflare Tunnel/Quick Tunnel, requests are proxied through Cloudflare's network. Cloudflare may process connection metadata consistent with their services and policies.

## Sharing
The builder does not sell your data. The API returns results only to the GPT session that made the request.

## Retention & deletion
Data persists on the operator machine until it is deleted by the operator. You can request deletion by contacting the operator.
Only the operator uses this GPT; do not use it unless you are the operator.

## Security
Access to the API is protected by a bearer token (VP_API_TOKEN). Do not share the token. Anyone with the token may be able to access the API while it is exposed.

## OpenAI processing
Your use of ChatGPT is also governed by OpenAI's own privacy policies, which are separate from this API's handling.

## Contact
For privacy questions or deletion requests, contact the GPT operator.
