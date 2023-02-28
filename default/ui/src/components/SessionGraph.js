import {CanvasJSChart} from 'canvasjs-react-charts'
import {useEffect, useState} from "react";

export default function SessionGraph({sessionId}) {
  const [options, updateOptions] = useState({})

  useEffect(() => {
    updateOptions({
      theme: "light1",
      animationEnabled: true,
      axisX: {},
      axisY: {},
      data: []
    })
  }, [])

  useEffect(() => {
    if (sessionId > 0) getTransactions();
    else {
      options.data = []
      updateOptions(options)
    }
  }, [sessionId])

  const getTransactions = () => {
    fetch(`/transactions?session_id=${sessionId}`, {})
      .then(response => {
        if (!response.ok) throw Error('Invalid response')
        else return response.json()
      })
      .then(json => {
        let maxY = 0, minY = 0
        console.log(json)
        let new_options = {
          theme: "light1",
          legend: {
            cursor: "pointer",
            itemclick: function (e) {
              //console.log("legend click: " + e.dataPointIndex);
              //console.log(e);
              if (typeof (e.dataSeries.visible) === "undefined" || e.dataSeries.visible) {
                e.dataSeries.visible = false;
              } else {
                e.dataSeries.visible = true;
              }

              e.chart.render();
            }
          },
          animationEnabled: true,
          axisX: {
            title: "steps"
          },
          axisY: {},
          data: []
        }
        for (const key of Object.keys(json)) {
          let values = json[key]
          let data = []
          console.log(key)
          console.log(values)
          for (let i = 0; i < values.length; i++) {
            let y_val = parseFloat(values[i])
            data.push({x: i + 1, y: y_val})
            minY = Math.min(minY, y_val)
            maxY = Math.max(maxY, y_val)
          }
          new_options.data.push({
            type: "line",
            name: key,
            markerType: "none",
            legendText: key,
            dataPoints: data,
            showInLegend: true
          })
        }
        new_options.axisY = {minimum: minY, maximum: maxY}
        updateOptions(new_options)
      })
  }

  return (
    <div className="sessionGraph">
      <CanvasJSChart options={options} />
    </div>
  )
}