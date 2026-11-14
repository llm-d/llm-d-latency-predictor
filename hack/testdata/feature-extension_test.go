/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package latencypredictorclient

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
)

const extensionFeature = "extension_test_size"

func extensionPredictor(t *testing.T) *Predictor {
	t.Helper()
	config := DefaultConfig()
	config.TrainingURL = os.Getenv("TRAINING_SERVER_URL")
	config.PredictionURLs = []string{os.Getenv("PREDICTION_SERVER_URL")}
	config.UseNativeXGBoost = false
	config.CoalesceWindow = 0
	config.FlushInterval = time.Hour
	predictor := New(config, logr.Discard())
	require.NoError(t, predictor.Start(t.Context()))
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		predictor.Stop(ctx)
	})
	_, err := predictor.GetMetrics(t.Context())
	require.NoError(t, err)
	return predictor
}

func TestFeatureExtensionTrainingIngest(t *testing.T) {
	predictor := extensionPredictor(t)
	entries := make([]TrainingEntry, 128)
	for index := range entries {
		size := 7 + index%64
		request, err := NewPredictionRequest(0, 1, 0, 0, 1, 0, WithFeature(extensionFeature, size))
		require.NoError(t, err)
		entries[index] = TrainingFromFeatures(request.protocolMessage().GetFeatures(), TrainingEntry{
			ActualTTFT: 10 + 2*float64(size), ActualTPOT: 5 + 0.5*float64(size), Timestamp: time.Now(),
		})
	}
	require.NoError(t, predictor.AddTrainingDataBulk(entries))
}

func TestFeatureExtensionModelRoundTrip(t *testing.T) {
	predictor := extensionPredictor(t)
	var requests []PredictionRequest
	for _, size := range []int{7, 70} {
		request, err := NewPredictionRequest(0, 1, 0, 0, 1, 0, WithFeature(extensionFeature, size))
		require.NoError(t, err)
		requests = append(requests, request)
	}
	strict, err := predictor.PredictBulkStrict(t.Context(), requests)
	require.NoError(t, err)
	require.Len(t, strict.Predictions, 2)
	require.Greater(t, strict.Predictions[1].TTFT, strict.Predictions[0].TTFT+1)
	require.Greater(t, strict.Predictions[1].TPOT, strict.Predictions[0].TPOT+1)
	plain, err := predictor.PredictBulk(t.Context(), requests)
	require.NoError(t, err)
	require.Len(t, plain.Predictions, 2)
	for index, request := range requests {
		single, predictErr := predictor.Predict(t.Context(), request)
		require.NoError(t, predictErr)
		require.InDelta(t, strict.Predictions[index].TTFT, single.TTFT, 0.001)
		require.InDelta(t, strict.Predictions[index].TPOT, single.TPOT, 0.001)
		require.InDelta(t, strict.Predictions[index].TTFT, plain.Predictions[index].TTFT, 0.001)
		require.InDelta(t, strict.Predictions[index].TPOT, plain.Predictions[index].TPOT, 0.001)
	}
	legacy, err := NewPredictionRequest(0, 1, 0, 0, 1, 0)
	require.NoError(t, err)
	missing, err := predictor.PredictBulkStrict(t.Context(), []PredictionRequest{legacy})
	require.NoError(t, err)
	require.Len(t, missing.Predictions, 1)
	require.InDelta(t, strict.Predictions[0].TTFT, missing.Predictions[0].TTFT, 0.001)
	require.InDelta(t, strict.Predictions[0].TPOT, missing.Predictions[0].TPOT, 0.001)
	if strict.Predictions[0].ModelType == bayesianRidgeModelType {
		coefficients, coefficientsErr := predictor.GetModelCoefficients(t.Context())
		require.NoError(t, coefficientsErr)
		require.InDelta(t, 2, coefficients.TTFTCoeffs[extensionFeature], 0.001)
		require.InDelta(t, 0.5, coefficients.TPOTCoeffs[extensionFeature], 0.001)
	}
	if strict.Predictions[0].ModelType == xgBoostModelType {
		trees, treesErr := predictor.getXGBoostTrees(t.Context())
		require.NoError(t, treesErr)
		for _, artifact := range []any{trees.TTFTTrees, trees.TPOTTrees} {
			data, marshalErr := json.Marshal(artifact)
			require.NoError(t, marshalErr)
			require.Contains(t, string(data), extensionFeature)
		}
	}
	if strict.Predictions[0].ModelType == gbmModelType {
		for _, target := range []string{"ttft", "tpot"} {
			request, requestErr := http.NewRequestWithContext(t.Context(), http.MethodGet, predictor.config.TrainingURL+"/model/"+target+"/lgb/importances", nil)
			require.NoError(t, requestErr)
			response, requestErr := predictor.httpClient.Do(request)
			require.NoError(t, requestErr)
			defer response.Body.Close()
			require.Equal(t, http.StatusOK, response.StatusCode)
			var importances map[string]float64
			require.NoError(t, json.NewDecoder(response.Body).Decode(&importances))
			require.Greater(t, importances[extensionFeature], float64(0))
		}
	}
}
