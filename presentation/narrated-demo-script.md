# Recorded walkthrough narration

Voice: Microsoft Zira Desktop, installed Windows synthetic voice (System.Speech), rate +1.
No voice cloning, external TTS, or paid service. Screens and training evidence are recorded.

## 00:00:00.000 — FedLoRA-Edge

This recorded walkthrough shows banking intent learning across five logical clients on one laptop.

The original frozen A LoRA benchmark is preserved. This confirmed demo uses frozen M P Net and a federated linear classifier head.

Source: Confirmed model: encoder-flower-balanced-v2 · selected version 500

## 00:00:14.750 — A banking question, classified locally

Here, recorded local inference maps, my card payment is still pending, to pending card payment.

The support response is team authored. Opening this screen does not start training.

Source: Actual browser recording · 14.0–29.0 seconds · functional example, not an accuracy test

## 00:00:29.750 — A locked, independent confirmation

On the locked independent confirmation set, the model scored ninety four percent: two hundred eighty two of three hundred.

The fixed clarification rule reached ninety eight point six five percent on answered queries, at seventy four point three three percent coverage.

Source: encoder-flower-balanced-v2 · pre-scoring lock commit 7413e6b · scored 3 November 2026

## 00:00:45.750 — When one client owns 90% of the data

Thirty three numerical federation development runs used reused validation.

With ninety percent of data on one client, sample weighting scored sixty two point five percent. Equal weights scored seventy six point five.

Adding clipping reached seventy eight point one seven, plus or minus zero point two nine percentage points.

Source: encoder-matrix-v1 · 400 training queries · 200 reused validation queries · fixed attempt 500

## 00:01:06.375 — A one-step client contributes useful knowledge

A one step verification client added eighteen percentage points in a matched inclusion comparison.

That client held four unique intents. This does not show that every weak device gives the same benefit.

Source: encoder-matrix-v1 · all clients 95.5% vs omit client 4 at 77.5% · reused validation, three seeds

## 00:01:19.250 — Dropout, quorum, and fresh rejoin

The recorded T L S Flower recovery run accepted five, four, five, two, then five clients.

On attempt four, quorum three was missed, so version three stayed unchanged.

Returning clients synchronized the current head. These were separate processes on one physical laptop.

Source: encoder-isolated-five-v8 · authenticated Flower Deployment Runtime · recorded recovery evidence

## 00:01:39.250 — A compact update. Clear next gates.

The head has sixty one thousand five hundred twenty raw F P thirty two bytes per client, per direction.

Physical laptop recovery and fresh human queries remain pending.

Source: 15,380 trained head parameters · no model, label, or threshold changed for this walkthrough
