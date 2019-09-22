import operator
import logging
import apache_beam as beam

class Transpose(beam.PTransform):
    """
    Class to transpose a large input matrix
    """

    def expand(self, pcoll):
        return (pcoll
            # Convert lines of text into individual words.
            | 'Set the input row keys' >> beam.ParDo(_SetRowKeys())
            | 'Set the new row keys' >> beam.ParDo(_SetColumnKeys())
            # Count the number of times each word occurs.
            | 'Collect the new rows' >> beam.GroupByKey()
            # Format each word and count into a printable string.
            | 'Transpose' >> beam.ParDo(_Transpose())
            | 'Printer' >> beam.ParDo(printer))


def printer(element):
    logging.info(element)


class _SetRowKeys(beam.DoFn):
    """ Set the orinal row, the first value of each row will be the new first row and the second values on each 
    row will be the second row. 
    """
    def process(self, element):
        row_key = element["row_key"]
        values = element["values"]
        return [{"row_key":row_key,"values":values}]


class _SetColumnKeys(beam.DoFn):
    """
    Outputing the values for each column in each row in order to create the transpose, the next step is 
    to do a group by. 
    """
    def process(self, element):
        row_key = element["row_key"]
        values = element["values"]
        for column_key,value in enumerate(values):
            yield (column_key,{row_key:value})


class _Transpose(beam.DoFn):
    """ After the groupby we need to sort the input to get the new rows in order."""
    def process(self, element):
        # element here is a list of all the grouped by the key i guess? 
        # This will be interesting to see, rest a bit now....
        row_key = element[0]
        values = element[1]
        output_values = []
        values = {k:v for element in values for k,v in element.items()}
        sorted_x = sorted(values.items(), key=operator.itemgetter(0))
        output_values = [values[1] for values in sorted_x] 
        return [(row_key,output_values)]





if __name__ == "__main__":
    main(pipeline_options=None)
    main_two(pipeline_options=None)