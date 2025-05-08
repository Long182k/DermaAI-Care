import {
  Body,
  Controller,
  Get,
  Param,
  Patch,
  Post,
  Query,
  UseGuards,
} from '@nestjs/common';
import { User } from '@prisma/client';
import { CurrentUser } from '../auth/@decorator/current-user.decorator';
import { JwtAuthGuard } from '../auth/@guard/jwt-auth.guard';
import { RolesGuard } from '../auth/@guard/roles.guard';
import { ParseISODatePipe } from '../common/pipes/parse-iso-date.pipe';
import { UpdateScheduleDto } from './dto/update-schedule.dto';
import { ScheduleTasksService } from './schedule-tasks.service';
import { ScheduleService } from './schedule.service';

@Controller('schedules')
@UseGuards(JwtAuthGuard, RolesGuard)
export class ScheduleController {
  constructor(
    private readonly scheduleService: ScheduleService,
    private readonly scheduleTasksService: ScheduleTasksService,
  ) {}

  @Get('doctor/:doctorId')
  async getDoctorSchedules(
    @Param('doctorId') doctorId: string,
    @Query('startDate', ParseISODatePipe) startDate: Date,
    @Query('endDate', ParseISODatePipe) endDate: Date,
    @Query('filter') filter?: 'day' | 'week' | 'month',
  ) {
    return await this.scheduleService.findDoctorSchedules(
      doctorId,
      startDate,
      endDate,
      filter,
    );
  }

  @Patch(':id')
  async updateSchedule(
    @Param('id') id: string,
    @Body() updateScheduleDto: UpdateScheduleDto,
    @CurrentUser() currentUser: User,
  ) {
    return await this.scheduleService.updateSchedule(
      id,
      updateScheduleDto,
      currentUser,
    );
  }

  @Post('generate')
  // @Roles(ROLE.ADMIN)
  async generateSchedules() {
    return await this.scheduleTasksService.generateSchedulesForDoctors();
  }
}
