package main

import (
	"encoding/csv"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"
)

type LinearRegression struct {
	slope     float64
	intercept float64
}

func (lr *LinearRegression) Fit(X, y []float64) {
	if len(X) != len(y) {
		panic("X and y must have the same length")
	}

	var sumX, sumY, sumXY, sumXSquare float64
	var mu sync.Mutex
	var wg sync.WaitGroup
	wg.Add(len(X))

	for i := 0; i < len(X); i++ {
		go func(i int) {
			mu.Lock()
			defer wg.Done()
			defer mu.Unlock()

			sumX += X[i]
			sumY += y[i]
			sumXY += X[i] * y[i]
			sumXSquare += X[i] * X[i]
		}(i)
	}
	wg.Wait()

	n := float64(len(X))

	lr.slope = (n*sumXY - sumX*sumY) / (n*sumXSquare - sumX*sumX)
	lr.intercept = (sumY - lr.slope*sumX) / n
}

func (lr *LinearRegression) Predict(X []float64) []float64 {
	predictions := make([]float64, len(X))
	for i := range X {
		predictions[i] = lr.slope*X[i] + lr.intercept
	}
	return predictions
}

func ReadDataset(url string) ([]float64, []float64) {

	var x []float64
	var y []float64

	resp, err := http.Get(url)
	if err != nil {
		fmt.Println("Error al hacer la solicitud HTTP: ", err)
		return x, y
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		fmt.Println("Error: no se pudo descargar el archivo CSV. Código de estado: ", resp.StatusCode)
		return x, y
	}

	reader := csv.NewReader(resp.Body)
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error al leer el archivo CSV:", err)
		return x, y
	}

	for i, record := range records {

		// Omitir primera linea
		if i == 0 {
			continue
		}

		xVal, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			fmt.Println("Error al convertir a float64:", err)
			return x, y
		}

		yVal, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			fmt.Println("Error al convertir a float64:", err)
			return x, y
		}

		x = append(x, xVal)
		y = append(y, yVal)
	}

	return x, y
}

func main() {
	X, y := ReadDataset("https://raw.githubusercontent.com/FrowsyFrog/T3_ProgramacionConcurrenteDistribuida/main/train.csv")

	// Medir tiempo de entrenamiento
	startTraining := time.Now()

	var lr LinearRegression
	lr.Fit(X, y)

	elapsedTraining := time.Since(startTraining)

	fmt.Printf("Training Time: %s\n", elapsedTraining)

	fmt.Printf("Slope: %.2f\n", lr.slope)
	fmt.Printf("Intercept: %.2f\n", lr.intercept)

	newX := []float64{100}
	predictions := lr.Predict(newX)

	fmt.Println("Input: ", newX)
	fmt.Println("Predictions:", predictions)
}
