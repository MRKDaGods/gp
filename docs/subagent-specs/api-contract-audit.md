# Frontend/Backend API Contract Audit

Date: 2026-05-27  
Scope: static analysis only; no dev server started; no fixes implemented.

## 1. Executive Verdict

The MTMC frontend/backend API surface is mostly wired after the redesign, but several gate-level contracts are not safe enough for the next Coder pass. The highest-risk break is `queryTimeline()` allowing a missing `runId` while the backend Pydantic model requires it, which can 422 the Stage 4 selected-track flow when only a gallery run is known. Kaggle dispatch exists and Stage 4 now sends the payload, but Kaggle error and cancel paths do not coherently mutate pipeline stage state or open the credentials modal. The static route inventory found 53 registered backend routes, 39 exported frontend API functions, 14 dead frontend functions, and 16 backend routes with no matching frontend consumer in the current UI surface.

## 2. 🔴 Critical Issues

1. **Stage 4 query can omit required `runId`, causing backend 422.**  
   Frontend `queryTimeline()` accepts `probeRunId: string | null | undefined` and only adds `body.runId` when truthy in [frontend/src/lib/api.ts](../../frontend/src/lib/api.ts#L646-L675). Backend `TimelineQueryRequest` requires `runId: str` in [backend/models/requests.py](../../backend/models/requests.py#L98-L103), and `/api/timeline/query` binds that model in [backend/routers/timeline.py](../../backend/routers/timeline.py#L22-L24). Timeline uses `effectiveGalleryRunId = galleryRunId ?? runId`, but calls `callQueryTimeline(runId ?? undefined)` in [frontend/src/components/stages/timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L623-L628), so a gallery-only state can send `galleryRunId` without `runId`.  
   Repro: load a preprocessed dataset/gallery without a probe `runId`, select tracklets, enter Stage 4.  
   Suggested fix: make frontend require/provide a probe run id before query mode, or relax backend `runId` to optional and resolve from `galleryRunId`/diagnostics deliberately.

2. **Kaggle 401/429 does not open the credentials modal and often degrades to `HTTP 401`/`HTTP 429`.**  
   Backend maps Kaggle auth, concurrency, and validation errors to 401/429/400 in [backend/routers/pipeline.py](../../backend/routers/pipeline.py#L154-L169). `fetchApi()` throws `ApiError(errorData.message || HTTP status)` and ignores FastAPI's common `detail` field in [frontend/src/lib/api.ts](../../frontend/src/lib/api.ts#L187-L207). Detection Stage catches errors as generic `err.message` in [frontend/src/components/stages/detection-stage.tsx](../../frontend/src/components/stages/detection-stage.tsx#L1005-L1020), and Upload's Stage 1 continue path does the same in [frontend/src/components/stages/upload-stage.tsx](../../frontend/src/components/stages/upload-stage.tsx#L348-L356). Inference has a better mapper in [frontend/src/components/stages/inference-stage.tsx](../../frontend/src/components/stages/inference-stage.tsx#L21-L31), but it still only returns text; it does not open credentials UI.  
   Repro: set Stage 1 or Stage 2/3 target to Kaggle with invalid credentials.  
   Suggested fix: centralize `ApiError` detail extraction and add a Kaggle-auth failure action that opens `KaggleCredentialsModal`.

3. **Kaggle cancel returns backend `cancelled`, but pipeline stage state remains running.**  
   Backend `POST /api/pipeline/kaggle-cancel/{run_id}` persists `status = "cancelled"` in [backend/routers/pipeline.py](../../backend/routers/pipeline.py#L319-L337). `KaggleStatusPanel.handleCancel()` calls `cancelKaggleKernel(runId)` but only updates local `cancelError/isCancelling` in [frontend/src/components/stages/kaggle-status-panel.tsx](../../frontend/src/components/stages/kaggle-status-panel.tsx#L140-L153). It does not call `usePipelineStore.updateStageProgress`, and `useKaggleStatus()` stores terminal status only inside the hook in [frontend/src/hooks/use-kaggle-status.ts](../../frontend/src/hooks/use-kaggle-status.ts#L61-L78).  
   Repro: dispatch any Kaggle stage, press Cancel kernel. The panel can show cancelled, but sidebar/header stage status can remain `running`.  
   Suggested fix: pass stage id or callback into `KaggleStatusPanel`, update the relevant stage to `idle` or `error/cancelled` equivalent, and clear global `isRunning`.

4. **Kaggle terminal completion is not propagated into `usePipelineStore` stage completion.**  
   Kaggle dispatch returns active run `status: queued` with `execution_target: "kaggle"` in [backend/routers/pipeline.py](../../backend/routers/pipeline.py#L171-L187). `RunStageWidget` swaps to `KaggleStatusPanel` when target is Kaggle and stage status is running in [frontend/src/components/pipeline/run/RunStageWidget.tsx](../../frontend/src/components/pipeline/run/RunStageWidget.tsx#L49-L87), but the panel/hook do not mutate pipeline stages on `complete/error/cancelled` in [frontend/src/hooks/use-kaggle-status.ts](../../frontend/src/hooks/use-kaggle-status.ts#L65-L78).  
   Repro: complete a Kaggle kernel successfully. Stage badge can remain `running` until some unrelated local polling refresh occurs.  
   Suggested fix: bridge Kaggle hook terminal statuses to `updateStageProgress(stage, ...)` and refresh imported/downloaded artifacts after completion.

5. **Stage 0 “Continue to Stage 1” bypasses the redesign execution target store.**  
   The redesigned Detection action reads `useStageExecutionStore` and constructs `{ kaggle: { target, username, key } }` in [frontend/src/components/stages/detection-stage.tsx](../../frontend/src/components/stages/detection-stage.tsx#L989-L1014). The Stage 0 action still calls `runStage(1, { videoId, config: { tracker } })` without consulting execution target or credentials in [frontend/src/components/stages/upload-stage.tsx](../../frontend/src/components/stages/upload-stage.tsx#L334-L350).  
   Repro: set Stage 1 target to Kaggle, return to Upload, click Continue to Stage 1. It starts local Stage 1.  
   Suggested fix: either remove this shortcut or route it through the same Stage 1 execution-target builder as Detection.

6. **Timeline helper uses a different default backend port than `api.ts`.**  
   Main API base defaults to `http://localhost:8000/api` in [frontend/src/lib/api.ts](../../frontend/src/lib/api.ts#L13-L13). Timeline hardcodes `http://localhost:8004/api` for matched clip URLs in [frontend/src/components/stages/timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L41-L41). Model registry service also defaults to `8004` in [frontend/src/services/models.ts](../../frontend/src/services/models.ts#L1-L1).  
   Repro: run backend on default 8000 with no `NEXT_PUBLIC_API_URL`; API calls work, but generated timeline/model URLs target 8004.  
   Suggested fix: import/reuse one frontend API base helper.

7. **`PipelineRunStatus` frontend type does not match single-stage backend responses.**  
   Frontend requires `runDir` and `stages` in [frontend/src/types/index.ts](../../frontend/src/types/index.ts#L172-L180), but single-stage backend `state.active_runs[run_id]` contains `stage`, `status`, `progress`, metadata, and no `stages`/`runDir` in [backend/routers/pipeline.py](../../backend/routers/pipeline.py#L113-L133). Most call sites cast `response.data as any`, e.g. [frontend/src/components/stages/inference-stage.tsx](../../frontend/src/components/stages/inference-stage.tsx#L302-L309), which hides the contract drift.  
   Repro: consume `runStage()` as typed without casts; TypeScript promises fields the backend does not return.  
   Suggested fix: split `PipelineRunStatus` into `SingleStageRunStatus` and `FullPipelineRunStatus`, or make fields optional and document the discriminator.

## 3. 🟡 Warnings

1. **`runFullPipeline()` is dead and request shape is misleading.** It sends a raw `config` object body in [frontend/src/lib/api.ts](../../frontend/src/lib/api.ts#L367-L373), while backend expects `PipelineRunRequest` with optional `config` field in [backend/routers/pipeline.py](../../backend/routers/pipeline.py#L218-L229). Zero call sites were found.
2. **`ApiError` is exported but detail handling is inconsistent.** `useKaggleStatus()` unwraps `data.detail` in [frontend/src/hooks/use-kaggle-status.ts](../../frontend/src/hooks/use-kaggle-status.ts#L21-L27), while many stage catches only read `err.message`.
3. **`uploadVideo()` and `importKaggleRunArtifacts()` XHR failures discard backend JSON details.** They reject `new ApiError('Upload failed', status)` and `new ApiError('Kaggle import failed', status)` in [frontend/src/lib/api.ts](../../frontend/src/lib/api.ts#L261-L276) and [frontend/src/lib/api.ts](../../frontend/src/lib/api.ts#L764-L780).
4. **`FusionConfigRequest` uses snake_case fields while backend aliases are camelCase.** Backend accepts both due `populate_by_name=True` in [backend/models/requests.py](../../backend/models/requests.py#L45-L56), but the public frontend contract in [frontend/src/lib/api.ts](../../frontend/src/lib/api.ts#L308-L320) differs from the ReID fusion endpoint's camelCase style.
5. **`KaggleRequestConfig` uses `dataset_slug`, while backend publishes alias `datasetSlug`.** Both are accepted by [backend/models/requests.py](../../backend/models/requests.py#L15-L28), but the UI never exposes dataset slug, so Kaggle dataset-input dispatch is untested from frontend.
6. **`getMatchedAlternatives()` returns raw non-`ApiResponse` while most API functions return `ApiResponse<T>`.** This is intentional normalization in [frontend/src/lib/api.ts](../../frontend/src/lib/api.ts#L530-L585), but it is easy to misuse.
7. **`getTrajectories()` trusts backend snake_case trajectory shape but returns `GlobalTrajectory[]` camelCase.** Backend returns raw `global_trajectories.json` in [backend/routers/tracklets.py](../../backend/routers/tracklets.py#L83-L89), while frontend `GlobalTrajectory` expects camelCase in [frontend/src/types/index.ts](../../frontend/src/types/index.ts#L60-L72). Timeline/output builders handle both in places, but the API type itself is optimistic.
8. **Search API has two dead wrappers over one backend route.** `searchByTracklet()` and `searchTracklet()` are unused in [frontend/src/lib/api.ts](../../frontend/src/lib/api.ts#L681-L710), while backend `/api/search/tracklet` remains active in [backend/routers/search.py](../../backend/routers/search.py#L16-L29).
9. **Confirm/reject tracklet is local-only and not persisted.** Timeline calls `confirmTrack/unconfirmTrack` in [frontend/src/components/stages/timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L1301-L1307), and the store mutates only local Zustand state in [frontend/src/store/index.ts](../../frontend/src/store/index.ts#L536-L562). Reloading loses decisions unless output filtering already generated artifacts.
10. **Apply alternative is local-only.** `handleApplyAlternative()` rewrites timeline rows via `applyTracksReplaceKeepingMeta(updated)` in [frontend/src/components/stages/timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L1211-L1292), with no backend mutation route.
11. **Backend route count is lower than request estimate.** Static decorator parsing found 53 routes, not ~57; aliases under `/pipeline/...` are included separately.
12. **Model registry is outside `api.ts`.** `/api/models` and `/api/models/{model_id}` are consumed through [frontend/src/services/models.ts](../../frontend/src/services/models.ts#L88-L123), not the audited `frontend/src/lib/api.ts` surface.

## 4. 🟢 Verified

- 25 frontend API functions have live call sites in `frontend/src/**`.
- 23 backend routes have direct `api.ts` wrapper coverage.
- 2 backend model registry routes have frontend coverage outside `api.ts`.
- Phase 1-7 fusion backend contract is present: top-level `fusion`, `resolve_pipeline_model()`, vehicle2/vehicle3 overrides, and `fusion_resolved` response fields are wired through [backend/routers/pipeline.py](../../backend/routers/pipeline.py#L87-L148) and [backend/services/pipeline_service.py](../../backend/services/pipeline_service.py#L170-L268).
- Phase 8-15 Kaggle backend routes exist: dispatch in [backend/routers/pipeline.py](../../backend/routers/pipeline.py#L154-L187), status in [backend/routers/pipeline.py](../../backend/routers/pipeline.py#L304-L317), cancel in [backend/routers/pipeline.py](../../backend/routers/pipeline.py#L319-L342), and job JSON write in [backend/services/kaggle_run_service.py](../../backend/services/kaggle_run_service.py#L34-L103).
- Stage 4 run paths now spread `stage4KaggleRequest()` into `runStage(4, ...)` at [frontend/src/components/stages/timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L689-L690), [frontend/src/components/stages/timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L776-L777), [frontend/src/components/stages/timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L907-L908), and [frontend/src/components/stages/timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L1322-L1323).
- State staleness is implemented: completing a stage stamps downstream completed stages with `staleSince` in [frontend/src/store/index.ts](../../frontend/src/store/index.ts#L106-L139), and `useStageState()` converts it to `stale` in [frontend/src/hooks/useStageState.ts](../../frontend/src/hooks/useStageState.ts#L16-L34).

## 5. Frontend API Inventory Table

| Export | Signature | Method + URL | Request schema | Response type | Error pattern | Store consumers |
|---|---|---|---|---|---|---|
| `singleCamReid` | [api.ts L213](../../frontend/src/lib/api.ts#L213-L218) | POST `/api/v1/reid/single_cam` | `SingleCamReIDRequestPayload`: `modelId`, `queries`, `gallery`, optional `topK/rerank/aqeK` | `SingleCamReIDResponsePayload` | `ApiError` via `fetchApi` | none; page local state |
| `fusionReid` | [api.ts L220](../../frontend/src/lib/api.ts#L220-L225) | POST `/api/v1/reid/fusion` | `FusionReIDRequestPayload`: `models`, `queries`, `gallery`, optional `topK/rerank/aqeK` | `FusionReIDResponsePayload` | `ApiError` | none; page local state |
| `submitEval` | [api.ts L227](../../frontend/src/lib/api.ts#L227-L232) | POST `/api/v1/eval/run` | `EvalRunRequestPayload`: `evalType`, optional `configOverrides` | `EvalJobResponsePayload` | `ApiError` | none; Eval page local state |
| `getEvalStatus` | [api.ts L234](../../frontend/src/lib/api.ts#L234-L236) | GET `/api/v1/eval/{jobId}/status` | path `jobId` | `EvalJobStatusPayload` | `ApiError` | none; Eval progress local state |
| `getEvalResult` | [api.ts L238](../../frontend/src/lib/api.ts#L238-L240) | GET `/api/v1/eval/{jobId}/result` | path `jobId` | `EvalJobResultPayload` | `ApiError` | none; Eval progress local state |
| `uploadVideo` | [api.ts L246](../../frontend/src/lib/api.ts#L246-L280) | POST `/api/videos/upload` | multipart `video` file | `ApiResponse<VideoFile>` normalized | XHR `ApiError`, detail discarded | `useVideoStore.addVideo`, `setCurrentVideo` |
| `getVideos` | [api.ts L282](../../frontend/src/lib/api.ts#L282-L290) | GET `/api/videos` | none | `ApiResponse<VideoFile[]>` normalized | `ApiError` | `useVideoStore.setVideos` |
| `getVideo` | [api.ts L292](../../frontend/src/lib/api.ts#L292-L298) | GET `/api/videos/{id}` | path `id` | `ApiResponse<VideoFile>` normalized | `ApiError` | output hydration local state |
| `deleteVideo` | [api.ts L300](../../frontend/src/lib/api.ts#L300-L302) | DELETE `/api/videos/{id}` | path `id` | `ApiResponse<void>` | `ApiError` | none |
| `runStage` | [api.ts L357](../../frontend/src/lib/api.ts#L357-L364) | POST `/api/pipeline/run-stage/{stage}` | `RunStageRequest`: optional `runId/videoId/cameraId/dataset/model_id/fusion/smokeTest/useCpu/config/kaggle` | `ApiResponse<PipelineRunStatus>` | `ApiError` | `usePipelineStore`, `useInferenceRunStore`, timeline local state |
| `runFullPipeline` | [api.ts L367](../../frontend/src/lib/api.ts#L367-L373) | POST `/api/pipeline/run` | raw config object, not `PipelineRunRequest` | `ApiResponse<PipelineRunStatus>` | `ApiError` | none |
| `getPipelineStatus` | [api.ts L376](../../frontend/src/lib/api.ts#L376-L380) | GET `/api/pipeline/status/{runId}` | path `runId` | `ApiResponse<PipelineRunStatus>` | `ApiError` | `usePipelineStore.updateStageProgress` in stage components |
| `cancelPipeline` | [api.ts L382](../../frontend/src/lib/api.ts#L382-L386) | POST `/api/pipeline/cancel/{runId}` | path `runId` | `ApiResponse<void>` | `ApiError` | stage actions set local idle |
| `getKaggleStatus` | [api.ts L388](../../frontend/src/lib/api.ts#L388-L390) | GET `/api/pipeline/kaggle-status/{runId}` | path `runId` | `ApiResponse<KaggleJobStatus>` | `ApiError`; hook unwraps detail | `useKaggleStatus` local hook state |
| `cancelKaggleKernel` | [api.ts L392](../../frontend/src/lib/api.ts#L392-L396) | POST `/api/pipeline/kaggle-cancel/{runId}` | path `runId` | `ApiResponse<KaggleJobStatus>` | `ApiError` | `KaggleStatusPanel` local state only |
| `getDetections` | [api.ts L402](../../frontend/src/lib/api.ts#L402-L415) | GET `/api/detections/{videoId}?frameId=` | path + optional query | `ApiResponse<Detection[]>` normalized | `ApiError` | `useDetectionStore.setDetections` |
| `getAllDetections` | [api.ts L417](../../frontend/src/lib/api.ts#L417-L430) | GET `/api/detections/{videoId}/all` | path `videoId` | `Map<number, Detection[]>` | `ApiError` | Detection local thumbnails/cache |
| `getFrameWithDetections` | [api.ts L432](../../frontend/src/lib/api.ts#L432-L460) | GET `/api/frames/{videoId}/{frameId}/detections` | path params | `ApiResponse<{frame,detections}>` normalized | `ApiError` | none |
| `extractFeatures` | [api.ts L467](../../frontend/src/lib/api.ts#L467-L475) | POST `/api/features/extract` | `{ trackletIds, cameraId }` | `ApiResponse<void>` | `ApiError` | none; backend route missing |
| `buildIndex` | [api.ts L477](../../frontend/src/lib/api.ts#L477-L482) | POST `/api/index/build/{runId}` | path `runId` | `ApiResponse<void>` | `ApiError` | none; backend route missing |
| `getTracklets` | [api.ts L487](../../frontend/src/lib/api.ts#L487-L496) | GET `/api/tracklets?cameraId=&videoId=` | optional query | `ApiResponse<Tracklet[]>` | `ApiError` | selection/timeline/output stores |
| `getMatchedSummary` | [api.ts L498](../../frontend/src/lib/api.ts#L498-L500) | GET `/api/runs/{runId}/matched_summary` | path `runId` | raw summary JSON | `ApiError` | timeline/output local state |
| `getMatchedAlternatives` | [api.ts L530](../../frontend/src/lib/api.ts#L530-L585) | GET `/api/runs/{runId}/matched_alternatives?...` | query `topK/anchor/exclude` | normalized `MatchedAlternativesPayload` | `ApiError` | timeline alternatives, refinement |
| `getMatchedAlternativeClipUrl` | [api.ts L587](../../frontend/src/lib/api.ts#L587-L594) | URL `/api/runs/{runId}/matched_alternatives/{path}` | path only | string URL | none | AlternativesSheet media |
| `getTrackletSequence` | [api.ts L612](../../frontend/src/lib/api.ts#L612-L625) | GET `/api/runs/{runId}/tracklet_sequence` | `cameraId`, `trackId`, `max_frames` | `TrackletSequencePayload` | `ApiError` | timeline grid, refinement |
| `getRunFullFrameUrl` | [api.ts L628](../../frontend/src/lib/api.ts#L628-L638) | URL `/api/runs/{runId}/full_frame` | query `cameraId/frameId` | string URL | none | timeline grid, refinement |
| `getTrajectories` | [api.ts L640](../../frontend/src/lib/api.ts#L640-L644) | GET `/api/trajectories/{runId}` | path `runId` | `ApiResponse<GlobalTrajectory[]>` | `ApiError` | timeline/output |
| `queryTimeline` | [api.ts L646](../../frontend/src/lib/api.ts#L646-L675) | POST `/api/timeline/query` | `videoId`, `selectedTrackIds`, optional `runId/galleryRunId/skipExports` | `ApiResponse<{stage4Available,...}>` | `ApiError` | timeline store |
| `searchByTracklet` | [api.ts L681](../../frontend/src/lib/api.ts#L681-L690) | POST `/api/search/tracklet` | `{ trackletId, cameraId, topK }` | `ApiResponse<SearchResult[]>` | `ApiError` | none; backend model mismatch |
| `searchTracklet` | [api.ts L692](../../frontend/src/lib/api.ts#L692-L710) | POST `/api/search/tracklet` | `{ trackletId, probeVideoId, galleryRunId, topK }` | `ApiResponse<ranked[]>` | `ApiError` | none |
| `getEvaluationResults` | [api.ts L713](../../frontend/src/lib/api.ts#L713-L717) | GET `/api/evaluation/{runId}` | path `runId` | `ApiResponse<EvaluationResult>` | `ApiError` | none |
| `generateSummaryVideo` | [api.ts L723](../../frontend/src/lib/api.ts#L723-L737) | POST `/api/visualization/summary/{runId}` | optional `{ includeClips, globalIds, speedup, dedupe }` | `ApiResponse<{videoUrl}>` | `ApiError` | output local state |
| `exportTrajectories` | [api.ts L739](../../frontend/src/lib/api.ts#L739-L744) | GET `/api/export/{runId}?format=` | query `json/csv/mot` | `ApiResponse<{downloadUrl}>` | `ApiError` | output export |
| `importKaggleRunArtifacts` | [api.ts L746](../../frontend/src/lib/api.ts#L746-L783) | POST `/api/runs/import-kaggle` | multipart `artifactsZip`, optional `runId/videoId/cameraId` | `ApiResponse<PipelineRunStatus>` | XHR `ApiError`, detail discarded | `usePipelineStore.setRunId/updateStageProgress`, `useVideoStore` |
| `getGovernorates/getCities/getZones/getCameras` | [api.ts L790](../../frontend/src/lib/api.ts#L790-L816) | GET location/camera endpoints | path/query | `ApiResponse<...>` | `ApiError` | none |
| `createWebSocket` | [api.ts L819](../../frontend/src/lib/api.ts#L819-L847) | WS `/api/ws/pipeline/{runId}` | path `runId` | `WebSocket` | callbacks only | none |
| `getFrameUrl/getVideoStreamUrl` | [api.ts L853](../../frontend/src/lib/api.ts#L853-L859) | URL helpers | path params | string URL | none | Detection video/frame media |
| `getDatasets` | [api.ts L887](../../frontend/src/lib/api.ts#L887-L889) | GET `/api/datasets` | none | `ApiResponse<DatasetFolder[]>` | `ApiError` | inference/output/dataset components |
| `processDataset` | [api.ts L891](../../frontend/src/lib/api.ts#L891-L897) | POST `/api/datasets/{folder}/process` | path `folder` | `ApiResponse<any>` | `ApiError` | dataset processing store/local state |
| `saveDatasetCameraCoordinates` | [api.ts L899](../../frontend/src/lib/api.ts#L899-L910) | PUT `/api/datasets/{folder}/camera-coordinates` | coordinate map | `ApiResponse<Record<...>>` | `ApiError` | none |

## 6. Backend Route Inventory Table

| Method | Full path | Handler | Request model/body | Response model | Dependencies/auth | Exceptions/status |
|---|---|---|---|---|---|---|
| GET | `/api/health` | [health_check](../../backend/routers/health.py#L9-L20) | none | raw dict | none | none explicit |
| GET | `/` | [root](../../backend/routers/health.py#L24-L32) | none | raw dict | none | none explicit |
| GET | `/api/locations/governorates` | [get_governorates](../../backend/routers/locations.py#L13-L24) | none | `ApiResponse` dict | none | none |
| GET | `/api/locations/cities/{governorate_id}` | [get_cities](../../backend/routers/locations.py#L27-L50) | path | `ApiResponse` dict | none | fallback to Cairo |
| GET | `/api/locations/zones/{city_id}` | [get_zones](../../backend/routers/locations.py#L53-L70) | path | `ApiResponse` dict | none | fallback to downtown |
| GET | `/api/cameras` | [get_cameras](../../backend/routers/locations.py#L73-L103) | optional `zoneId` | `ApiResponse` dict | `AppState` | none |
| POST | `/api/videos/upload` | [upload_video](../../backend/routers/videos.py#L25-L46) | multipart `video` | raw `ApiResponse<video>` | `AppState` | 500 catch-all |
| GET | `/api/videos` | [get_videos](../../backend/routers/videos.py#L49-L55) | none | raw `ApiResponse<videos>` | `AppState` | none |
| GET | `/api/videos/{video_id}` | [get_video](../../backend/routers/videos.py#L58-L66) | path | raw `ApiResponse<video>` | `AppState` | 404 video |
| DELETE | `/api/videos/{video_id}` | [delete_video](../../backend/routers/videos.py#L69-L79) | path | raw `ApiResponse<None>` | `AppState` | 404 video |
| GET | `/api/videos/stream/{video_id}` | [stream_video](../../backend/routers/videos.py#L82-L106) | path | `FileResponse` | `AppState` | 404 video/file |
| GET | `/api/detections/{video_id}` | [get_detections](../../backend/routers/detections.py#L22-L48) | path, optional `frameId` | raw `ApiResponse<detections>` | `AppState` | 404 video |
| GET | `/api/detections/{video_id}/all` | [get_all_detections](../../backend/routers/detections.py#L51-L84) | path | raw grouped dict | `AppState` | 404 video |
| GET | `/api/frames/{video_id}/{frame_id}/detections` | [get_frame_with_detections](../../backend/routers/frames.py#L25-L45) | path | raw frame+detections | `AppState` | inherited from detections |
| GET | `/api/frames/{video_id}/{frame_id}` | [get_frame_image](../../backend/routers/frames.py#L48-L84) | path | image stream/file | `AppState` | 500 cv2, 404 video/file/frame |
| GET | `/api/crops/{video_id}` | [get_crop](../../backend/routers/crops.py#L134-L184) | query bbox/frame | image stream | `AppState` | 500 cv2, 404 video/file/frame, 400 bbox |
| GET | `/api/crops/run/{run_id}` | [get_crop_from_run](../../backend/routers/crops.py#L189-L247) | query camera/bbox/frame | image stream | none | 500 cv2/read, 404 frames, 400 bbox |
| GET | `/api/tracklets` | [get_tracklets](../../backend/routers/tracklets.py#L19-L80) | optional `cameraId/videoId` | raw tracklet summaries | `AppState` | 404 video |
| GET | `/api/trajectories/{run_id}` | [get_trajectories](../../backend/routers/tracklets.py#L83-L89) | path | raw trajectories | none | empty success if missing |
| GET | `/api/runs/{run_id}/matched_summary` | [get_matched_summary](../../backend/routers/runs.py#L36-L44) | path | raw JSON | none | 404 missing summary |
| GET | `/api/runs/{run_id}/matched_clips/{filename}` | [get_matched_clip](../../backend/routers/runs.py#L47-L93) | path | `FileResponse` | none | 400 filename, 404 clip |
| GET | `/api/runs/{run_id}/matched_alternatives` | [get_matched_alternatives](../../backend/routers/runs.py#L106-L306) | query topK/anchor/exclude | raw alternatives | none | 400/403 run id, 404 summary |
| GET | `/api/runs/{run_id}/matched_alternatives/{clip_relpath:path}` | [get_matched_alternative_clip](../../backend/routers/runs.py#L309-L330) | path | `FileResponse` | none | 400 filename, 404 clip |
| POST | `/api/runs/import-kaggle` | [import_kaggle_run_artifacts](../../backend/routers/runs.py#L333-L425) | multipart `artifactsZip`, form ids | raw `PipelineRunStatus` dict | `AppState`, config flag | 403 disabled, 400 zip, 404 video |
| GET | `/api/runs/{run_id}/full_frame` | [get_run_full_frame](../../backend/routers/frames.py#L87-L110) | query `cameraId/frameId` | image file | none | 400 camera, 404 frame |
| GET | `/api/runs/{run_id}/tracklet_sequence` | [get_tracklet_sequence](../../backend/routers/frames.py#L113-L209) | query `cameraId/trackId/max_frames` | raw sequence | none | 400 params, 404 run/tracklet |
| GET | `/api/evaluation/{run_id}` | [get_evaluation_results](../../backend/routers/results.py#L15-L55) | path | raw metrics | none | 404 run |
| POST | `/api/visualization/summary/{run_id}` | [generate_summary_video](../../backend/routers/results.py#L58-L118) | optional JSON config | raw `{ videoUrl }` | `AppState` | 404 run |
| GET | `/api/export/{run_id}` | [export_trajectories](../../backend/routers/export.py#L14-L80) | query `format` | raw `{ downloadUrl }` | none | 400 format, 404 run |
| GET | `/api/download/{run_id}/{filename}` | [download_export_file](../../backend/routers/export.py#L83-L110) | path | file | none | 404 run/file |
| GET | `/api/datasets` | [list_datasets](../../backend/routers/datasets.py#L56-L140) | none | raw dataset list | `AppState` | empty success if missing dir |
| POST | `/api/datasets/{folder}/process` | [process_dataset](../../backend/routers/datasets.py#L145-L180) | path | raw run dict | `BackgroundTasks`, `AppState` | 404 folder |
| PUT | `/api/datasets/{folder}/camera-coordinates` | [put_camera_coordinates](../../backend/routers/datasets.py#L183-L209) | JSON coordinate dict | raw coordinate dict | none | 400 traversal, 404 folder |
| GET | `/api/models` | [get_models](../../backend/routers/models.py#L15-L31) | query filters | `ModelListResponse` | none | Pydantic validation |
| GET | `/api/models/{model_id}` | [get_model_entry](../../backend/routers/models.py#L34-L41) | path | `ModelDetailResponse` | none | 404 model |
| POST | `/api/v1/reid/single_cam` | [single_cam_reid](../../backend/routers/reid.py#L43-L59) | `SingleCamReIDRequest` | `SingleCamReIDResponse` | lazy ReID service | 503 deps, service mapped errors |
| POST | `/api/v1/reid/fusion` | [fusion_reid](../../backend/routers/reid.py#L63-L87) | `FusionReIDRequest` | `FusionReIDResponse` | lazy ReID service | 503 deps, service mapped errors |
| POST | `/api/v1/eval/run` | [run_eval](../../backend/routers/eval.py#L24-L32) | `EvalRunRequest` | `EvalJobResponse` | background jobs | 422 unsupported/invalid |
| GET | `/api/v1/eval/{job_id}/status` | [eval_status](../../backend/routers/eval.py#L35-L40) | path | `EvalJobStatusResponse` | job service | 404 job |
| GET | `/api/v1/eval/{job_id}` | [eval_status_alias](../../backend/routers/eval.py#L43-L45) | path | `EvalJobStatusResponse` | job service | 404 via alias |
| GET | `/api/v1/eval/{job_id}/result` | [eval_result](../../backend/routers/eval.py#L48-L64) | path | `EvalJobResultResponse` | job service | 404, 409, 500 |
| POST | `/api/pipeline/run-stage/{stage}` | [run_stage](../../backend/routers/pipeline.py#L66-L216) | `PipelineRunRequest` | raw run dict | `BackgroundTasks`, `AppState`, Kaggle services | 400, 401, 404, 429, 500 |
| POST | `/pipeline/run-stage/{stage}` | [run_stage alias](../../backend/routers/pipeline.py#L66-L67) | same | same | same | same |
| POST | `/api/pipeline/run` | [run_full_pipeline](../../backend/routers/pipeline.py#L218-L293) | `PipelineRunRequest` | raw full run dict | `BackgroundTasks`, `AppState` | 400 model/config |
| GET | `/api/pipeline/status/{run_id}` | [get_pipeline_status](../../backend/routers/pipeline.py#L296-L301) | path | raw active run | `AppState` | 404 run |
| GET | `/api/pipeline/kaggle-status/{run_id}` | [get_kaggle_status](../../backend/routers/pipeline.py#L304-L317) | path | raw Kaggle job | Kaggle service | 404, 401, 400 |
| GET | `/pipeline/kaggle-status/{run_id}` | [kaggle status alias](../../backend/routers/pipeline.py#L304-L305) | same | same | same | same |
| POST | `/api/pipeline/kaggle-cancel/{run_id}` | [cancel_kaggle](../../backend/routers/pipeline.py#L319-L342) | path | raw Kaggle job | Kaggle service | 404, 401, 500 |
| POST | `/pipeline/kaggle-cancel/{run_id}` | [cancel alias](../../backend/routers/pipeline.py#L319-L320) | same | same | same | same |
| POST | `/api/pipeline/cancel/{run_id}` | [cancel_pipeline](../../backend/routers/pipeline.py#L344-L351) | path | raw `None` | `AppState` | 404 run |
| WEBSOCKET | `/api/ws/pipeline/{run_id}` | [websocket_pipeline_updates](../../backend/routers/pipeline.py#L353-L371) | path | websocket JSON | `AppState` | logs close errors |
| POST | `/api/search/tracklet` | [search_by_tracklet](../../backend/routers/search.py#L16-L145) | `SearchRequest` | raw ranked results | `AppState` | 400 preconditions, 500 search |
| POST | `/api/timeline/query` | [query_timeline](../../backend/routers/timeline.py#L22-L74) | `TimelineQueryRequest` | raw timeline response | `AppState` | 404 video, service ValueErrors become 500 unless internal handled |

## 7. Cross-Reference Table

| Frontend fn | Method | URL | Backend route | Request match | Response match | Call sites post-redesign | Status |
|---|---|---|---|---|---|---|---|
| `singleCamReid` | POST | `/v1/reid/single_cam` | `/api/v1/reid/single_cam` | ✅ camel aliases | ✅ response aliases | [reid/page.tsx](../../frontend/src/app/reid/page.tsx#L41-L49) | ✅ verified |
| `fusionReid` | POST | `/v1/reid/fusion` | `/api/v1/reid/fusion` | ✅ | ✅ | [fusion/page.tsx](../../frontend/src/app/fusion/page.tsx#L61-L68) | ✅ verified |
| `submitEval` | POST | `/v1/eval/run` | `/api/v1/eval/run` | ✅ | ✅ | [eval/page.tsx](../../frontend/src/app/eval/page.tsx#L35-L35) | ✅ verified |
| `getEvalStatus` | GET | `/v1/eval/{id}/status` | `/api/v1/eval/{id}/status` | ✅ | ✅ | [EvalProgressPanel.tsx](../../frontend/src/components/eval/EvalProgressPanel.tsx#L42-L46) | ✅ verified |
| `getEvalResult` | GET | `/v1/eval/{id}/result` | `/api/v1/eval/{id}/result` | ✅ | ✅ | [EvalProgressPanel.tsx](../../frontend/src/components/eval/EvalProgressPanel.tsx#L46-L46) | ✅ verified |
| `uploadVideo` | POST | `/videos/upload` | `/api/videos/upload` | ✅ multipart `video` | ✅ normalized | [upload-stage.tsx](../../frontend/src/components/stages/upload-stage.tsx#L69-L78) | ✅ verified |
| `getVideos` | GET | `/videos` | `/api/videos` | ✅ | ✅ normalized | [upload-stage.tsx](../../frontend/src/components/stages/upload-stage.tsx#L47-L48) | ✅ verified |
| `getVideo` | GET | `/videos/{id}` | `/api/videos/{video_id}` | ✅ | ✅ normalized | [output-stage.tsx](../../frontend/src/components/stages/output-stage.tsx#L587-L590) | ✅ verified |
| `deleteVideo` | DELETE | `/videos/{id}` | `/api/videos/{video_id}` | ✅ | ✅ | none | 🆕 frontend-only dead wrapper |
| `runStage` | POST | `/pipeline/run-stage/{stage}` | `/api/pipeline/run-stage/{stage}` | ⚠️ nullable timeline `runId`; type drift | ⚠️ single-stage shape drift | 11 matches, e.g. [inference-stage.tsx](../../frontend/src/components/stages/inference-stage.tsx#L288-L309) | ⚠️ minor/critical by flow |
| `runFullPipeline` | POST | `/pipeline/run` | `/api/pipeline/run` | ❌ raw config body vs model wrapper | ⚠️ | none | 🆕 frontend-only dead/broken stub |
| `getPipelineStatus` | GET | `/pipeline/status/{runId}` | `/api/pipeline/status/{run_id}` | ✅ | ⚠️ raw single/full run union | stage polling in [detection-stage.tsx](../../frontend/src/components/stages/detection-stage.tsx#L399-L433), [inference-stage.tsx](../../frontend/src/components/stages/inference-stage.tsx#L85-L105), [timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L697-L715) | ✅ verified |
| `cancelPipeline` | POST | `/pipeline/cancel/{runId}` | `/api/pipeline/cancel/{run_id}` | ✅ | ✅ | stage cancel actions | ✅ verified |
| `getKaggleStatus` | GET | `/pipeline/kaggle-status/{runId}` | `/api/pipeline/kaggle-status/{run_id}` | ✅ | ✅ | [use-kaggle-status.ts](../../frontend/src/hooks/use-kaggle-status.ts#L70-L78) | ⚠️ no store mutation |
| `cancelKaggleKernel` | POST | `/pipeline/kaggle-cancel/{runId}` | `/api/pipeline/kaggle-cancel/{run_id}` | ✅ | ✅ | [kaggle-status-panel.tsx](../../frontend/src/components/stages/kaggle-status-panel.tsx#L147-L147) | ⚠️ no store mutation |
| `getDetections` | GET | `/detections/{videoId}` | `/api/detections/{video_id}` | ✅ | ✅ normalized | [detection-stage.tsx](../../frontend/src/components/stages/detection-stage.tsx#L328-L328) | ✅ verified |
| `getAllDetections` | GET | `/detections/{videoId}/all` | `/api/detections/{video_id}/all` | ✅ | ✅ Map normalized | [detection-stage.tsx](../../frontend/src/components/stages/detection-stage.tsx#L319-L319) | ✅ verified |
| `getFrameWithDetections` | GET | `/frames/{videoId}/{frameId}/detections` | `/api/frames/{video_id}/{frame_id}/detections` | ✅ | ✅ normalized | none | 🆕 dead wrapper |
| `extractFeatures` | POST | `/features/extract` | none | ❌ | ❌ | none | 🆕 frontend-only stub |
| `buildIndex` | POST | `/index/build/{runId}` | none | ❌ | ❌ | none | 🆕 frontend-only stub |
| `getTracklets` | GET | `/tracklets` | `/api/tracklets` | ✅ | ⚠️ TS tracklet shape optimistic | selection/timeline/output | ✅ verified |
| `getMatchedSummary` | GET | `/runs/{runId}/matched_summary` | same | ✅ | raw | timeline/output | ✅ verified |
| `getMatchedAlternatives` | GET | `/runs/{runId}/matched_alternatives` | same | ✅ | ✅ normalized | timeline/refinement | ✅ verified |
| `getMatchedAlternativeClipUrl` | URL | `/runs/{runId}/matched_alternatives/{path}` | same | ✅ | file URL | AlternativesSheet | ✅ verified |
| `getTrackletSequence` | GET | `/runs/{runId}/tracklet_sequence` | same | ✅ | ✅ | timeline grid/refinement | ✅ verified |
| `getRunFullFrameUrl` | URL | `/runs/{runId}/full_frame` | same | ✅ | file URL | timeline grid/refinement | ✅ verified |
| `getTrajectories` | GET | `/trajectories/{runId}` | `/api/trajectories/{run_id}` | ✅ | ⚠️ snake/camel | timeline/output | ⚠️ minor mismatch |
| `queryTimeline` | POST | `/timeline/query` | `/api/timeline/query` | ❌ `runId` optional frontend vs required backend | ✅ envelope | [timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L608-L614) | ❌ broken edge flow |
| `searchByTracklet` | POST | `/search/tracklet` | `/api/search/tracklet` | ❌ sends `cameraId`; backend ignores and needs `probeVideoId/galleryRunId` | response mismatch | none | 🆕 dead/broken wrapper |
| `searchTracklet` | POST | `/search/tracklet` | `/api/search/tracklet` | ✅ | ✅ | none | 🆕 dead wrapper |
| `getEvaluationResults` | GET | `/evaluation/{runId}` | same | ✅ | ✅ | none | 🆕 dead wrapper |
| `generateSummaryVideo` | POST | `/visualization/summary/{runId}` | same | ✅ partial; backend only uses `includeClips` | ✅ | output | ✅ verified |
| `exportTrajectories` | GET | `/export/{runId}` | same | ✅ | ✅ | output | ✅ verified |
| `importKaggleRunArtifacts` | POST | `/runs/import-kaggle` | same | ✅ multipart | ✅ | upload | ✅ verified |
| Location wrappers | GET | `/locations/*`, `/cameras` | same | ✅ | ✅ | none | 🆕 dead wrappers |
| `createWebSocket` | WS | `/ws/pipeline/{runId}` | `/api/ws/pipeline/{run_id}` | ✅ with API_BASE | raw websocket | none | 🆕 dead wrapper |
| URL media helpers | URL | `/frames`, `/videos/stream` | same | ✅ | media | detection | ✅ verified |
| Dataset wrappers | GET/POST/PUT | `/datasets*` | same | ✅ | ✅ | `getDatasets/processDataset` used; coordinate save unused | ✅/🆕 mixed |

Backend-only with no current matching frontend consumer: `/`, `/api/health`, `/api/locations/*`, `/api/cameras`, `/api/crops/{video_id}`, `/api/crops/run/{run_id}`, `/api/download/{run_id}/{filename}` as a direct fetch wrapper, `/api/v1/eval/{job_id}` alias, `/pipeline/*` non-API aliases, `/api/ws/pipeline/{run_id}`, `/api/evaluation/{run_id}`, and `/api/datasets/{folder}/camera-coordinates`.

## 8. Critical Flows Verification

1. **Upload video (Stage 0): ✅ verified**  
   Click/drop path: drop/select calls `handleFiles()` in [upload-stage.tsx](../../frontend/src/components/stages/upload-stage.tsx#L63-L91), `uploadVideo()` posts multipart to `/api/videos/upload` in [api.ts](../../frontend/src/lib/api.ts#L246-L280), backend stores the video in `state.uploaded_videos` in [videos.py](../../backend/routers/videos.py#L25-L44), then frontend calls `addVideo()` and `setCurrentVideo()` in [upload-stage.tsx](../../frontend/src/components/stages/upload-stage.tsx#L74-L78). Store mutation is [store/index.ts](../../frontend/src/store/index.ts#L244-L250). Error reaches `ErrorBanner` via `uploadError` in [upload-stage.tsx](../../frontend/src/components/stages/upload-stage.tsx#L83-L85) and [upload-stage.tsx](../../frontend/src/components/stages/upload-stage.tsx#L186-L186).

2. **Import Kaggle artifacts (Stage 0): ✅ verified with warning**  
   Select zip in [upload-stage.tsx](../../frontend/src/components/stages/upload-stage.tsx#L116-L160), call `importKaggleRunArtifacts()` multipart in [api.ts](../../frontend/src/lib/api.ts#L746-L783), backend route materializes outputs and sets `state.active_runs[run_id]` plus video mapping in [runs.py](../../backend/routers/runs.py#L333-L425). Frontend sets `runId` and marks Stage 6 completed in [upload-stage.tsx](../../frontend/src/components/stages/upload-stage.tsx#L138-L145). Warning: XHR error path discards backend detail in [api.ts](../../frontend/src/lib/api.ts#L771-L780).

3. **Run Stage 1 Detection: ⚠️ split behavior**  
   Detection redesign action reads target and credentials in [detection-stage.tsx](../../frontend/src/components/stages/detection-stage.tsx#L989-L1003), sends `runStage(1, { videoId, config, kaggle })` in [detection-stage.tsx](../../frontend/src/components/stages/detection-stage.tsx#L1005-L1013), backend validates `videoId` unless Kaggle dataset input in [pipeline.py](../../backend/routers/pipeline.py#L93-L105), dispatches Kaggle and writes `kaggle_job.json` through [pipeline.py](../../backend/routers/pipeline.py#L154-L187) and [kaggle_run_service.py](../../backend/services/kaggle_run_service.py#L92-L103). Break: Upload Stage's `Continue to Stage 1` calls local-only `runStage(1, { videoId, config })` in [upload-stage.tsx](../../frontend/src/components/stages/upload-stage.tsx#L334-L350), bypassing `useStageExecutionStore`.

4. **Stage 1 polling: ✅ local verified, ⚠️ Kaggle divergent**  
   Local polling calls `getPipelineStatus(runId)` in [detection-stage.tsx](../../frontend/src/components/stages/detection-stage.tsx#L399-L433), backend returns `state.active_runs[run_id]` in [pipeline.py](../../backend/routers/pipeline.py#L296-L301), and frontend marks completed/error via `updateStageProgress` in [detection-stage.tsx](../../frontend/src/components/stages/detection-stage.tsx#L411-L428). Kaggle status polling is separate through `KaggleStatusPanel` and does not update `usePipelineStore` terminal state.

5. **Run Stage 2 feature extraction with fusion: ✅ contract verified, ⚠️ Kaggle terminal gap**  
   Fusion payload is built from store with top-level `fusion` in [inference-stage.tsx](../../frontend/src/components/stages/inference-stage.tsx#L111-L122), Stage 2 request includes `dataset`, `model_id`, `fusion`, `kaggle`, and `config.filters` in [inference-stage.tsx](../../frontend/src/components/stages/inference-stage.tsx#L288-L301). Backend `PipelineRunRequest` accepts top-level `fusion/kaggle` in [requests.py](../../backend/models/requests.py#L82-L94), resolves fusion overrides in [pipeline.py](../../backend/routers/pipeline.py#L87-L132), and stores `fusion_resolved`. Frontend stores response metadata in [inference-stage.tsx](../../frontend/src/components/stages/inference-stage.tsx#L302-L309). Break is Kaggle terminal state not bridged.

6. **Run Stage 3 index build: ✅ contract verified, ⚠️ same terminal gap**  
   Same `runBackendStage(3)` path as Stage 2 in [inference-stage.tsx](../../frontend/src/components/stages/inference-stage.tsx#L280-L333). Backend dispatches `execute_stage(... stage=3 ...)` or Kaggle with `[3]` in [pipeline.py](../../backend/routers/pipeline.py#L154-L205). Local `pollStageStatus()` marks completed in [inference-stage.tsx](../../frontend/src/components/stages/inference-stage.tsx#L85-L105).

7. **Run Stage 4 Association: ✅ Phase 8 Kaggle payload sent, ⚠️ query/runId break**  
   `stage4KaggleRequest()` returns `{ kaggle: { target: "kaggle", username, key } }` in [timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L91-L96). All run paths spread it into `runStage(4, ...)` at [timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L689-L690), [timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L776-L777), [timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L907-L908), and [timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L1322-L1323). Backend dispatches Stage 4 in [pipeline.py](../../backend/routers/pipeline.py#L154-L205). Break: selected query path can call `/timeline/query` without required `runId` as described in Critical Issue 1.

8. **Confirm/reject tracklet: ⚠️ local-only**  
   Toggle calls `handleConfirmToggle()` in [timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L1301-L1307), which calls Zustand `confirmTrack/unconfirmTrack`. Store only mutates `confirmedTracks`, `timelineClipFilterEngaged`, and row `confirmed` flags in [store/index.ts](../../frontend/src/store/index.ts#L536-L562). No backend route persists this. Suggested fix: document local-only semantics or add a run-scoped confirmation mutation.

9. **Apply alternative: ⚠️ local-only**  
   Alternatives are loaded by `getMatchedAlternatives()` in [timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L1167-L1184). Applying an alternative rewrites the selected timeline row and calls `applyTracksReplaceKeepingMeta(updated)` in [timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L1211-L1292). No `applyAlternative()` backend route exists. Suggested fix: either persist this as a timeline edit artifact or keep it explicitly local and make output generation use current local tracks.

10. **Refinement re-search: ✅ local mutation verified**  
   Button/event dispatch is in [refinement-stage.tsx](../../frontend/src/components/stages/refinement-stage.tsx#L218-L224). Handler calls `getMatchedAlternatives(runId, anchor...)` per selected anchor in [refinement-stage.tsx](../../frontend/src/components/stages/refinement-stage.tsx#L153-L167), aggregates alternatives, and mutates timeline store with `replaceTracksSyncingRowFlags(refinedTracks)` in [refinement-stage.tsx](../../frontend/src/components/stages/refinement-stage.tsx#L182-L202). Backend route is [runs.py](../../backend/routers/runs.py#L106-L306). No backend mutation is expected.

11. **Generate summary video: ✅ verified, ⚠️ no polling endpoint**  
   Output builds `summaryVideoPayload` from timeline selections in [output-stage.tsx](../../frontend/src/components/stages/output-stage.tsx#L407-L416), calls `generateSummaryVideo()` on download in [output-stage.tsx](../../frontend/src/components/stages/output-stage.tsx#L521-L538), and also fire-and-forgets generation during load in [output-stage.tsx](../../frontend/src/components/stages/output-stage.tsx#L605-L612). Backend `POST /api/visualization/summary/{run_id}` returns a direct `videoUrl` synchronously in [results.py](../../backend/routers/results.py#L58-L118). The requested poll/download flow does not exist as separate status/download APIs; download uses `/api/download/{run_id}/{filename}` returned by backend.

12. **Kaggle cancel: ❌ backend works, UI store incomplete**  
   Button calls `cancelKaggleKernel(runId)` in [kaggle-status-panel.tsx](../../frontend/src/components/stages/kaggle-status-panel.tsx#L140-L153). Backend cancels/persists status in [pipeline.py](../../backend/routers/pipeline.py#L319-L337). Missing: pipeline `stage.status` is not set to cancelled/idle/error, and `isRunning` is not cleared by the panel.

## 9. Type Contract Spot-Checks

| Endpoint | Frontend type | Backend model/return | Verdict |
|---|---|---|---|
| POST `/api/pipeline/run-stage/{stage}` | `RunStageRequest` [api.ts](../../frontend/src/lib/api.ts#L344-L355) | `PipelineRunRequest` [requests.py](../../backend/models/requests.py#L82-L94) | ⚠️ fields accepted; response type drift |
| Fusion payload inside run-stage | `FusionConfigRequest` snake_case [api.ts](../../frontend/src/lib/api.ts#L308-L320) | `FusionConfig` aliases [requests.py](../../backend/models/requests.py#L45-L56) | ✅ accepted via `populate_by_name`; public style mismatch |
| Kaggle payload inside run-stage | `KaggleRequestConfig` [api.ts](../../frontend/src/lib/api.ts#L322-L327) | `KaggleConfig` [requests.py](../../backend/models/requests.py#L15-L28) | ✅ target/creds align; `dataset_slug` alias mismatch tolerated |
| GET `/api/pipeline/kaggle-status/{run_id}` | `KaggleJobStatus` [api.ts](../../frontend/src/lib/api.ts#L329-L342) | job JSON state [kaggle_run_service.py](../../backend/services/kaggle_run_service.py#L92-L103) | ✅ fields align except optional backend `message` on terminal cancel |
| POST `/api/videos/upload` | `ApiResponse<VideoFile>` normalized [api.ts](../../frontend/src/lib/api.ts#L15-L29) | raw `_video_payload` [videos.py](../../backend/routers/videos.py#L17-L23) | ✅ normalization covers `latestRunId` |
| GET `/api/tracklets` | `Tracklet[]` [types/index.ts](../../frontend/src/types/index.ts#L43-L55) | summary rows with `id`, `startFrame`, `representativeBbox` [tracklets.py](../../backend/routers/tracklets.py#L42-L76) | ⚠️ API return is not true `Tracklet`; consumers use `any`/summary fields |
| GET `/api/trajectories/{run_id}` | `GlobalTrajectory[]` camelCase [types/index.ts](../../frontend/src/types/index.ts#L60-L72) | raw JSON from pipeline [tracklets.py](../../backend/routers/tracklets.py#L83-L89) | ⚠️ snake/camel mixed; builders compensate |
| POST `/api/timeline/query` | optional `runId` body [api.ts](../../frontend/src/lib/api.ts#L646-L675) | required `TimelineQueryRequest.runId` [requests.py](../../backend/models/requests.py#L98-L103) | ❌ optional/required mismatch |
| POST `/api/search/tracklet` | `searchByTracklet` sends `cameraId` [api.ts](../../frontend/src/lib/api.ts#L681-L690) | `SearchRequest` has `probeVideoId/galleryRunId/trackletId/topK` [requests.py](../../backend/models/requests.py#L106-L111) | ❌ one wrapper broken; `searchTracklet` aligns |
| POST `/api/visualization/summary/{run_id}` | optional `includeClips/globalIds/speedup/dedupe` [api.ts](../../frontend/src/lib/api.ts#L723-L737) | backend only reads `includeClips` [results.py](../../backend/routers/results.py#L68-L75) | ⚠️ extra fields ignored |

## 10. Dead/Orphan Inventory

**Frontend functions in `api.ts` with zero call sites (14):** `deleteVideo`, `runFullPipeline`, `getFrameWithDetections`, `extractFeatures`, `buildIndex`, `searchByTracklet`, `searchTracklet`, `getEvaluationResults`, `getGovernorates`, `getCities`, `getZones`, `getCameras`, `createWebSocket`, `saveDatasetCameraCoordinates`. Counts came from static scan of `frontend/src/**/*.{ts,tsx}` excluding `api.ts`.

**Backend routes with no current matching frontend consumer (16):** `/`, `/api/health`, `/api/locations/governorates`, `/api/locations/cities/{governorate_id}`, `/api/locations/zones/{city_id}`, `/api/cameras`, `/api/crops/{video_id}`, `/api/crops/run/{run_id}`, `/api/v1/eval/{job_id}` alias, `/pipeline/run-stage/{stage}` alias, `/pipeline/kaggle-status/{run_id}` alias, `/pipeline/kaggle-cancel/{run_id}` alias, `/api/ws/pipeline/{run_id}`, `/api/evaluation/{run_id}`, `/api/datasets/{folder}/camera-coordinates`, `/api/download/{run_id}/{filename}` as a direct wrapper. Note: `/api/download` is indirectly used through backend-returned URLs.

**TS types/interfaces likely stale or misleading:** `Tracklet` as returned by `getTracklets()` ([types/index.ts](../../frontend/src/types/index.ts#L43-L55)), `GlobalTrajectory` as returned by `getTrajectories()` ([types/index.ts](../../frontend/src/types/index.ts#L60-L72)), `PipelineRunStatus` for single-stage runs ([types/index.ts](../../frontend/src/types/index.ts#L172-L185)), and location/camera response types attached to dead wrappers ([api.ts](../../frontend/src/lib/api.ts#L790-L816).

## 11. Error Path Audit

| Endpoint/flow | Backend statuses | Frontend handling | User-visible result | Verdict |
|---|---|---|---|---|
| Upload video | 500 in [videos.py](../../backend/routers/videos.py#L44-L46) | XHR generic error [api.ts](../../frontend/src/lib/api.ts#L261-L276), `ErrorBanner` [upload-stage.tsx](../../frontend/src/components/stages/upload-stage.tsx#L186-L186) | visible but generic | ⚠️ |
| Import Kaggle artifacts | 403/400/404 in [runs.py](../../backend/routers/runs.py#L343-L368) | XHR generic error [api.ts](../../frontend/src/lib/api.ts#L771-L780), `ErrorBanner` [upload-stage.tsx](../../frontend/src/components/stages/upload-stage.tsx#L151-L154) | visible but generic | ⚠️ |
| Run Stage Kaggle | 401/429/400/500 in [pipeline.py](../../backend/routers/pipeline.py#L154-L215) | Detection generic [detection-stage.tsx](../../frontend/src/components/stages/detection-stage.tsx#L1015-L1020), Inference mapped text [inference-stage.tsx](../../frontend/src/components/stages/inference-stage.tsx#L21-L31) | no credentials modal; 429 only meaningful in inference | ❌ |
| Pipeline status | 404 in [pipeline.py](../../backend/routers/pipeline.py#L296-L301) | stage poll catches to error [timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L1330-L1343) | visible stage error | ✅ |
| Kaggle status | 404/401/400 in [pipeline.py](../../backend/routers/pipeline.py#L304-L317) | 404 suppressed, others shown in panel [use-kaggle-status.ts](../../frontend/src/hooks/use-kaggle-status.ts#L80-L94) | visible for 401/400; 404 silent | ⚠️ |
| Kaggle cancel | 404/401/500 in [pipeline.py](../../backend/routers/pipeline.py#L319-L342) | local `cancelError` [kaggle-status-panel.tsx](../../frontend/src/components/stages/kaggle-status-panel.tsx#L147-L153) | visible in panel; store not updated | ⚠️ |
| Timeline query | 404 video in [timeline.py](../../backend/routers/timeline.py#L25-L29), 422 Pydantic possible | catch updates Stage 4 error [timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L1015-L1024) | visible `formatNetworkFailure`; root runId issue remains | ❌ |
| Alternatives | 400/403/404 in [runs.py](../../backend/routers/runs.py#L106-L122) | catches and shows friendly alternatives error [timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx#L1185-L1197) | visible | ✅ |
| Refinement re-search | same alternatives statuses | catches to `ErrorBanner` [refinement-stage.tsx](../../frontend/src/components/stages/refinement-stage.tsx#L203-L207) | visible | ✅ |
| Summary video | 404 in [results.py](../../backend/routers/results.py#L64-L67) | fire-and-forget swallows; download handler lacks banner [output-stage.tsx](../../frontend/src/components/stages/output-stage.tsx#L521-L538) | possibly silent during download | ⚠️ |

## 12. State Coherence Findings

- `useStageState(stage)` derives `blocked` from previous stage `completed` or progress >= 100 in [useStageState.ts](../../frontend/src/hooks/useStageState.ts#L16-L34). This matches the current `StageProgress[]` shape.
- `staleSince` triggers when a completed upstream stage completes after downstream completed stages; implemented in [store/index.ts](../../frontend/src/store/index.ts#L106-L139) and interpreted in [useStageState.ts](../../frontend/src/hooks/useStageState.ts#L24-L28). ✅
- `lastRunAt` and `completedAt` are stamped in `updateStageProgress()` when a stage starts/completes in [store/index.ts](../../frontend/src/store/index.ts#L113-L124). ✅
- Sidebar dots and workspace badges both consume `useStageState()`/`toStageStatus()`: sidebar in [main-dashboard.tsx](../../frontend/src/components/layout/main-dashboard.tsx#L133-L182), StageShell contract in [main-dashboard.tsx](../../frontend/src/components/layout/main-dashboard.tsx#L210-L244), header badge in [PipelineRunHeader.tsx](../../frontend/src/components/pipeline/header/PipelineRunHeader.tsx#L71-L118). ✅
- ContractBanner blocked pill resolves to previous stage through `blockedBy` and `onNavigateToStage` in [main-dashboard.tsx](../../frontend/src/components/layout/main-dashboard.tsx#L210-L244), rendered as a button in [ContractBanner.tsx](../../frontend/src/components/pipeline/shell/ContractBanner.tsx#L83-L93). ✅
- Coherence gap: Kaggle terminal/cancel states live in `useKaggleStatus()` and never stamp `completedAt`, `lastRunAt`, `staleSince`, or `isRunning` in the pipeline store. ❌
- Coherence gap: `flushPipelineFromStage(4)` resets downstream stages and timeline/refinement data before Stage 2/3 runs in [inference-stage.tsx](../../frontend/src/components/stages/inference-stage.tsx#L270-L278), but Stage 4 local-only alternative edits are not represented as stage state changes. ⚠️

## 13. Fix Work Plan

1. **Commit 1: centralize API error detail and Kaggle auth UX.**  
   Files: [frontend/src/lib/api.ts](../../frontend/src/lib/api.ts), [frontend/src/components/stages/detection-stage.tsx](../../frontend/src/components/stages/detection-stage.tsx), [frontend/src/components/stages/inference-stage.tsx](../../frontend/src/components/stages/inference-stage.tsx), [frontend/src/components/pipeline/run/ExecutionTargetToggle.tsx](../../frontend/src/components/pipeline/run/ExecutionTargetToggle.tsx).  
   Add `ApiError.fromResponse`/detail extraction, use it in XHR paths, and open credentials modal on 401. Preserve 429 text.

2. **Commit 2: fix Stage 4 timeline query contract.**  
   Files: [frontend/src/lib/api.ts](../../frontend/src/lib/api.ts), [frontend/src/components/stages/timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx), [backend/models/requests.py](../../backend/models/requests.py), [backend/routers/timeline.py](../../backend/routers/timeline.py).  
   Decide one contract: either always require a probe run id before calling query, or make backend `runId` optional and resolve/fail with a domain 400 message instead of Pydantic 422.

3. **Commit 3: bridge Kaggle status into pipeline state.**  
   Files: [frontend/src/components/pipeline/run/RunStageWidget.tsx](../../frontend/src/components/pipeline/run/RunStageWidget.tsx), [frontend/src/components/stages/kaggle-status-panel.tsx](../../frontend/src/components/stages/kaggle-status-panel.tsx), [frontend/src/hooks/use-kaggle-status.ts](../../frontend/src/hooks/use-kaggle-status.ts), [frontend/src/store/index.ts](../../frontend/src/store/index.ts).  
   Pass stage id/callback to the panel and stamp `completed/error/idle`, `completedAt`, and `isRunning` on terminal statuses.

4. **Commit 4: unify Stage 1 run action.**  
   Files: [frontend/src/components/stages/upload-stage.tsx](../../frontend/src/components/stages/upload-stage.tsx), [frontend/src/components/stages/detection-stage.tsx](../../frontend/src/components/stages/detection-stage.tsx), possibly new helper under `frontend/src/lib/`.  
   Ensure Upload `Continue to Stage 1` and Detection `Run Stage 1` use the same execution target and Kaggle credential builder.

5. **Commit 5: normalize frontend API base.**  
   Files: [frontend/src/lib/api.ts](../../frontend/src/lib/api.ts), [frontend/src/components/stages/timeline-stage.tsx](../../frontend/src/components/stages/timeline-stage.tsx), [frontend/src/services/models.ts](../../frontend/src/services/models.ts).  
   Export one API base/origin helper and remove hardcoded `8004` defaults.

6. **Commit 6: split or repair run status types.**  
   Files: [frontend/src/types/index.ts](../../frontend/src/types/index.ts), [frontend/src/lib/api.ts](../../frontend/src/lib/api.ts), stage consumers.  
   Model `SingleStageRunStatus` separately from full pipeline status; remove `as any` where possible.

7. **Commit 7: prune or implement dead wrappers.**  
   Files: [frontend/src/lib/api.ts](../../frontend/src/lib/api.ts), backend routers as needed.  
   Remove dead wrappers with no planned UI (`extractFeatures`, `buildIndex`, old location API), or wire them into explicit UI routes/tests. Keep model registry in `services/models.ts` but align API base.
