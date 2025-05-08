import { Controller, Get } from '@nestjs/common';
import { StatisticsService } from './statistics.service';
import { ApiTags, ApiOperation, ApiResponse } from '@nestjs/swagger';

@ApiTags('Statistics')
@Controller('statistics')
export class StatisticsController {
  constructor(private readonly statisticsService: StatisticsService) {}

  @Get('overview')
  @ApiOperation({ summary: 'Get overview statistics' })
  @ApiResponse({ 
    status: 200, 
    description: 'Returns overview statistics for dashboard visualization' 
  })
  getOverview() {
    return this.statisticsService.getOverview();
  }

  @Get('patients')
  @ApiOperation({ summary: 'Get patient statistics' })
  @ApiResponse({ 
    status: 200, 
    description: 'Returns patient statistics for dashboard visualization' 
  })
  getPatients() {
    return this.statisticsService.getPatients();
  }

  @Get('doctors')
  @ApiOperation({ summary: 'Get doctor statistics' })
  @ApiResponse({ 
    status: 200, 
    description: 'Returns doctor statistics for dashboard visualization' 
  })
  getDoctors() {
    return this.statisticsService.getDoctors();
  }

  @Get('predictions')
  @ApiOperation({ summary: 'Get prediction statistics' })
  @ApiResponse({ 
    status: 200, 
    description: 'Returns prediction statistics for dashboard visualization' 
  })
  getPredictions() {
    return this.statisticsService.getPredictions();
  }

  @Get('appointments')
  @ApiOperation({ summary: 'Get appointment statistics' })
  @ApiResponse({ 
    status: 200, 
    description: 'Returns appointment statistics for dashboard visualization' 
  })
  getAppointments() {
    return this.statisticsService.getAppointments();
  }
}
