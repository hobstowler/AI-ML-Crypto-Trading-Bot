import {CanvasJSChart} from 'canvasjs-react-charts'

export default function SessionGraph(session, transactions) {
  // formats data for consumption by canvasJS graph and corrects the timestamp (ms to s)
  const formatData = (data) => {
    let dataPoints = []
    for (let i = 0; i < data.length; i++) {
      dataPoints.push({
        x: new Date(data[i].x * 1000),
        y: data[i].y
      })
    }
    return dataPoints
  }

  // builds graph options
  const buildOptions = () => {
    let newOptions = {
      theme: "light1",
      animationEnabled: true,
      axisX: {
        valueFormatString: "DD-MMM"
      },
      axisY: {
        prefix: "$",
        title: "Price (in USD)",
        includeZero: false
      },
      data: [{
        type: "candlestick",
        yValueFormatString: "$###0.00",
        xValueType: "dateTime",
        dataPoints: null
      }]
    }
  }

  return (
    <div className="sessionGraph">

    </div>
  )
}